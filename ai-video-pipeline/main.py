#!/usr/bin/env python3
"""
AI Video Generation Pipeline
=============================
Entry points
------------
  python main.py                        # Launch Gradio UI (default)
  python main.py --topic "black holes"  # Headless CLI generation
  python main.py --help
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_ui(host: str = "0.0.0.0", port: int = 7860, share: bool = False) -> None:
    from ui import create_app
    app = create_app()
    app.launch(server_name=host, server_port=port, share=share)


def run_cli(
    topic: str,
    style: str = "engaging and informative",
    upload_yt: bool = False,
) -> None:
    import config
    from utils.helpers import slug, ensure_ffmpeg
    from pipeline import (
        ScriptGenerator, NarrationGenerator, MusicGenerator,
        ImageGenerator, VideoAnimator, SubtitleGenerator, VideoAssembler,
    )

    ensure_ffmpeg()

    import uuid
    job_id = str(uuid.uuid4())[:8]
    job_dir = config.OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    logger.info("═" * 60)
    logger.info("Topic: %s", topic)
    logger.info("Job ID: %s", job_id)
    logger.info("═" * 60)

    # ── Pipeline stages ───────────────────────────────────────────────────────

    logger.info("[1/7] Generating script …")
    script = ScriptGenerator().generate(topic, style)
    logger.info("  Title: %s", script.title)

    narrations = [s.narration for s in script.scenes]
    image_prompts = [s.image_prompt for s in script.scenes]
    durations = [s.duration for s in script.scenes]

    logger.info("[2/7] Synthesising narration (%d scenes) …", len(narrations))
    scene_audio = NarrationGenerator().generate_scene_audio(narrations, job_dir / "audio")
    narration_combined = job_dir / "narration_full.wav"
    _concat_wav_cli(scene_audio, narration_combined)

    logger.info("[3/7] Generating background music …")
    music_path = job_dir / "music.wav"
    MusicGenerator().generate(script.music_prompt, music_path, duration=int(sum(durations)) + 5)

    logger.info("[4/7] Generating images …")
    image_paths = ImageGenerator().generate_batch(image_prompts, job_dir / "images")

    logger.info("[5/7] Animating images …")
    clip_paths = VideoAnimator().animate_batch(image_paths, job_dir / "clips", durations=durations)

    logger.info("[6/7] Generating subtitles …")
    sub_path = job_dir / "subtitles.ass"
    SubtitleGenerator().generate_ass(narration_combined, sub_path)

    logger.info("[7/7] Assembling final video …")
    output_path = config.VIDEOS_DIR / f"{slug(script.title)}_{job_id}.mp4"
    VideoAssembler().assemble(
        clips=clip_paths,
        narration_audio=narration_combined,
        music_audio=music_path,
        subtitle_path=sub_path,
        output_path=output_path,
        title=script.title,
    )

    logger.info("═" * 60)
    logger.info("Video ready: %s", output_path)
    logger.info("═" * 60)

    # ── Optional uploads ──────────────────────────────────────────────────────

    if upload_yt:
        from uploaders import YouTubeUploader
        vid_id = YouTubeUploader().upload(
            output_path, title=script.title,
            description=script.description, tags=script.hashtags,
        )
        logger.info("YouTube Shorts: https://youtube.com/shorts/%s", vid_id)


def _concat_wav_cli(paths: list[Path], output: Path) -> None:
    import subprocess
    list_file = output.parent / "wav_list.txt"
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in paths))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(list_file), "-c", "copy", str(output)],
        capture_output=True, timeout=60, check=True,
    )


# ── CLI argument parsing ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Video Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--topic", help="Video topic (enables headless CLI mode)")
    parser.add_argument("--style", default="engaging and informative",
                        help="Narration style")
    parser.add_argument("--upload-youtube", action="store_true",
                        help="Upload result to YouTube Shorts")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio server host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true",
                        help="Create public Gradio share link")

    args = parser.parse_args()

    if args.topic:
        run_cli(
            topic=args.topic,
            style=args.style,
            upload_yt=args.upload_youtube,
        )
    else:
        run_ui(host=args.host, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
