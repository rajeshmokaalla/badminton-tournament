"""FFmpeg-based video assembly: combine clips, audio, music, and subtitles."""

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import config
from utils.helpers import timed

logger = logging.getLogger(__name__)


class VideoAssembler:

    # ── Public API ────────────────────────────────────────────────────────────

    @timed("assembly")
    def assemble(
        self,
        clips: list[Path],
        narration_audio: Path,
        music_audio: Path,
        subtitle_path: Path | None,
        output_path: Path,
        title: str = "",
    ) -> Path:
        """
        Combine animated clips + narration + music + subtitles → final MP4.

        Steps
        -----
        1. Concatenate all video clips into one stream.
        2. Mix narration (loud) + music (quiet) into stereo audio.
        3. Burn-in ASS subtitles if available.
        4. Write final 9:16 MP4.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            concat_video = tmp / "concat.mp4"
            mixed_audio = tmp / "mixed.aac"
            raw_video = tmp / "raw.mp4"

            # ── Step 1: concatenate clips ──────────────────────────────────
            self._concat_clips(clips, concat_video)

            # ── Step 2: mix narration + music ─────────────────────────────
            self._mix_audio(narration_audio, music_audio, mixed_audio)

            # ── Step 3: merge video + audio ───────────────────────────────
            self._merge_av(concat_video, mixed_audio, raw_video)

            # ── Step 4: burn subtitles ────────────────────────────────────
            if subtitle_path and Path(subtitle_path).exists() and Path(subtitle_path).stat().st_size > 10:
                self._burn_subtitles(raw_video, subtitle_path, output_path)
            else:
                import shutil
                shutil.copy2(raw_video, output_path)

        logger.info("Final video: %s", output_path)
        return output_path

    def get_duration(self, path: Path) -> float:
        """Return duration of a media file in seconds via ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            return 0.0
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _concat_clips(self, clips: list[Path], output: Path) -> None:
        list_file = output.parent / "concat_list.txt"
        list_file.write_text(
            "\n".join(f"file '{c.resolve()}'" for c in clips),
            encoding="utf-8",
        )
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-vf", f"scale={config.IMAGE_WIDTH}:{config.IMAGE_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={config.IMAGE_WIDTH}:{config.IMAGE_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
            "-an",
            str(output),
        ]
        self._run(cmd, "concat clips")

    def _mix_audio(
        self,
        narration: Path,
        music: Path,
        output: Path,
        music_volume: float = 0.15,
    ) -> None:
        # Narration at full volume; music ducked to music_volume
        cmd = [
            "ffmpeg", "-y",
            "-i", str(narration),
            "-i", str(music),
            "-filter_complex",
            f"[0:a]volume=1.0[nar];[1:a]volume={music_volume}[mus];"
            "[nar][mus]amix=inputs=2:duration=first:dropout_transition=2[out]",
            "-map", "[out]",
            "-c:a", "aac", "-b:a", "192k",
            str(output),
        ]
        self._run(cmd, "mix audio")

    def _merge_av(self, video: Path, audio: Path, output: Path) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-i", str(audio),
            "-c:v", "copy",
            "-c:a", "copy",
            "-shortest",
            str(output),
        ]
        self._run(cmd, "merge AV")

    def _burn_subtitles(self, video: Path, subs: Path, output: Path) -> None:
        subs_abs = str(subs.resolve()).replace("\\", "/").replace(":", "\\:")

        if subs.suffix == ".ass":
            vf = f"ass={subs_abs}"
        else:
            vf = (
                f"subtitles={subs_abs}:force_style='"
                "FontName=Arial Rounded MT Bold,FontSize=72,"
                "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                "BackColour=&H80000000,Bold=1,Outline=3,Shadow=1,"
                "Alignment=2,MarginV=80'"
            )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "copy",
            str(output),
        ]
        self._run(cmd, "burn subtitles")

    @staticmethod
    def _run(cmd: list[str], label: str) -> None:
        logger.debug("FFmpeg [%s]: %s", label, " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            logger.error("FFmpeg [%s] stderr:\n%s", label, result.stderr.decode())
            raise RuntimeError(f"FFmpeg step '{label}' failed (rc={result.returncode})")
