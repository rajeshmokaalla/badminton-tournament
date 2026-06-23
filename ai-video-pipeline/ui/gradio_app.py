"""Gradio chat interface for the AI video generation pipeline."""

import logging
import threading
import uuid
from pathlib import Path

import gradio as gr

import config
from utils.helpers import slug

logger = logging.getLogger(__name__)


def _build_pipeline():
    """Lazy import pipeline to avoid loading heavy models at UI startup."""
    from pipeline import (
        ScriptGenerator,
        NarrationGenerator,
        MusicGenerator,
        ImageGenerator,
        VideoAnimator,
        SubtitleGenerator,
        VideoAssembler,
    )
    return {
        "script": ScriptGenerator(),
        "narration": NarrationGenerator(),
        "music": MusicGenerator(),
        "image": ImageGenerator(),
        "animator": VideoAnimator(),
        "subtitle": SubtitleGenerator(),
        "assembler": VideoAssembler(),
    }


_pipeline_cache: dict = {}
_lock = threading.Lock()


def get_pipeline() -> dict:
    with _lock:
        if not _pipeline_cache:
            _pipeline_cache.update(_build_pipeline())
    return _pipeline_cache


# ── Core generation function ──────────────────────────────────────────────────

def generate_video(
    topic: str,
    style: str,
    upload_yt: bool,
    upload_ig: bool,
    ig_video_url: str,
    progress=gr.Progress(track_tqdm=True),
) -> tuple:
    """
    Full pipeline: topic → finished MP4 (+ optional uploads).
    Returns (status_message, video_path_or_None, log_text).
    """
    if not topic.strip():
        return "Please enter a topic.", None, ""

    job_id = str(uuid.uuid4())[:8]
    job_dir = config.OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str):
        logger.info(msg)
        log_lines.append(msg)

    try:
        pl = get_pipeline()

        # ── 1. Script ─────────────────────────────────────────────────────
        progress(0.05, desc="Writing script …")
        log(f"[1/7] Generating script for: {topic!r}")
        script = pl["script"].generate(topic, style)
        log(f"      Title: {script.title}")
        log(f"      Scenes: {len(script.scenes)}")

        narrations = [s.narration for s in script.scenes]
        image_prompts = [s.image_prompt for s in script.scenes]
        durations = [s.duration for s in script.scenes]

        # ── 2. Narration ──────────────────────────────────────────────────
        progress(0.15, desc="Generating narration …")
        log("[2/7] Synthesising narration …")
        scene_audio_paths = pl["narration"].generate_scene_audio(
            narrations, job_dir / "audio"
        )
        # Concatenate scene audio into one file for Whisper + mixing
        narration_combined = job_dir / "narration_full.wav"
        _concat_wav(scene_audio_paths, narration_combined)
        log(f"      Audio: {narration_combined}")

        # ── 3. Music ──────────────────────────────────────────────────────
        progress(0.25, desc="Generating background music …")
        log("[3/7] Generating music …")
        music_path = job_dir / "music.wav"
        total_dur = int(sum(durations)) + 5
        pl["music"].generate(script.music_prompt, music_path, duration=total_dur)
        log(f"      Music: {music_path}")

        # ── 4. Images ─────────────────────────────────────────────────────
        progress(0.40, desc="Generating images …")
        log(f"[4/7] Generating {len(image_prompts)} images …")
        image_paths = pl["image"].generate_batch(
            image_prompts, job_dir / "images", base_name="scene"
        )
        log(f"      Images: {len(image_paths)}")

        # ── 5. Animation ──────────────────────────────────────────────────
        progress(0.55, desc="Animating images …")
        log("[5/7] Animating images …")
        clip_paths = pl["animator"].animate_batch(
            image_paths, job_dir / "clips", durations=durations
        )
        log(f"      Clips: {len(clip_paths)}")

        # ── 6. Subtitles ──────────────────────────────────────────────────
        progress(0.70, desc="Generating subtitles …")
        log("[6/7] Generating subtitles …")
        sub_path = job_dir / "subtitles.ass"
        pl["subtitle"].generate_ass(narration_combined, sub_path)
        log(f"      Subtitles: {sub_path}")

        # ── 7. Assemble ───────────────────────────────────────────────────
        progress(0.85, desc="Assembling final video …")
        log("[7/7] Assembling final video …")
        output_path = config.VIDEOS_DIR / f"{slug(script.title)}_{job_id}.mp4"
        pl["assembler"].assemble(
            clips=clip_paths,
            narration_audio=narration_combined,
            music_audio=music_path,
            subtitle_path=sub_path,
            output_path=output_path,
            title=script.title,
        )
        log(f"      Output: {output_path}")

        # ── Uploads ───────────────────────────────────────────────────────
        upload_results = []
        if upload_yt:
            progress(0.92, desc="Uploading to YouTube …")
            try:
                from uploaders import YouTubeUploader
                yt = YouTubeUploader()
                vid_id = yt.upload(
                    output_path,
                    title=script.title,
                    description=script.description,
                    tags=script.hashtags,
                )
                url = f"https://youtube.com/shorts/{vid_id}"
                upload_results.append(f"YouTube: {url}")
                log(f"YouTube uploaded: {url}")
            except Exception as e:
                upload_results.append(f"YouTube FAILED: {e}")
                log(f"YouTube error: {e}")

        if upload_ig and ig_video_url.strip():
            progress(0.96, desc="Uploading to Instagram …")
            try:
                from uploaders import InstagramUploader
                ig = InstagramUploader()
                caption = f"{script.title}\n\n" + " ".join(f"#{t}" for t in script.hashtags)
                media_id = ig.upload_reel(ig_video_url.strip(), caption=caption)
                upload_results.append(f"Instagram media_id: {media_id}")
                log(f"Instagram uploaded: {media_id}")
            except Exception as e:
                upload_results.append(f"Instagram FAILED: {e}")
                log(f"Instagram error: {e}")

        progress(1.0, desc="Done!")
        status = f"Video ready: {output_path.name}"
        if upload_results:
            status += "\n" + "\n".join(upload_results)

        return status, str(output_path), "\n".join(log_lines)

    except Exception as exc:
        logger.exception("Pipeline error")
        return f"Error: {exc}", None, "\n".join(log_lines)


def _concat_wav(paths: list[Path], output: Path) -> None:
    """Concatenate WAV files using ffmpeg."""
    import subprocess, tempfile
    list_file = output.parent / "wav_list.txt"
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in paths))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(list_file), "-c", "copy", str(output)],
        capture_output=True, timeout=60, check=True,
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="AI Video Pipeline",
        theme=gr.themes.Soft(primary_hue="violet"),
        css=".output-video { max-height: 600px; }",
    ) as app:
        gr.Markdown(
            """
            # AI Video Generation Pipeline
            Generate viral YouTube Shorts & Instagram Reels from any topic.

            **Stack:** Llama/Qwen · Piper TTS · MusicGen · SDXL/FLUX · Whisper · FFmpeg
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                topic_input = gr.Textbox(
                    label="Topic",
                    placeholder="e.g. 'The mystery of black holes' or '5 life-changing Python tips'",
                    lines=2,
                )
                style_input = gr.Dropdown(
                    label="Style",
                    choices=[
                        "engaging and informative",
                        "humorous and entertaining",
                        "dramatic and cinematic",
                        "educational and clear",
                        "motivational and inspiring",
                    ],
                    value="engaging and informative",
                )

                with gr.Accordion("Upload settings (optional)", open=False):
                    upload_yt = gr.Checkbox(label="Upload to YouTube Shorts", value=False)
                    upload_ig = gr.Checkbox(label="Upload to Instagram Reels", value=False)
                    ig_video_url = gr.Textbox(
                        label="Public video URL for Instagram",
                        placeholder="https://…/video.mp4  (required for Instagram)",
                        info="Meta API requires a public HTTPS URL to fetch the video.",
                    )

                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

            with gr.Column(scale=3):
                status_output = gr.Textbox(label="Status", lines=3, interactive=False)
                video_output = gr.Video(label="Generated Video", elem_classes="output-video")
                log_output = gr.Textbox(
                    label="Pipeline Log", lines=12, interactive=False, max_lines=20
                )

        generate_btn.click(
            fn=generate_video,
            inputs=[topic_input, style_input, upload_yt, upload_ig, ig_video_url],
            outputs=[status_output, video_output, log_output],
        )

        gr.Markdown(
            """
            ### Notes
            - First run downloads model weights (~2–10 GB depending on backend).
            - YouTube upload requires `credentials/youtube_client_secrets.json`.
            - Instagram upload requires `INSTAGRAM_ACCESS_TOKEN` and `INSTAGRAM_ACCOUNT_ID` in `.env`.
            - Set `IMAGE_BACKEND=flux` in `.env` for higher quality (needs more VRAM).
            - Enable `SVD_ENABLED=true` for video animation via Stable Video Diffusion.
            """
        )

    return app
