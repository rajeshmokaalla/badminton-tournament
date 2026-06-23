"""Subtitle generation using OpenAI Whisper (local, offline)."""

import logging
from pathlib import Path

import config
from utils.helpers import timed

logger = logging.getLogger(__name__)


class SubtitleGenerator:
    def __init__(self, model_size: str = config.WHISPER_MODEL):
        self.model_size = model_size
        self._model = None

    # ── Public API ────────────────────────────────────────────────────────────

    @timed("subtitles")
    def generate_srt(self, audio_path: Path, output_path: Path) -> Path:
        """Transcribe *audio_path* and write an SRT file to *output_path*."""
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Transcribing %s with Whisper %s …", audio_path.name, self.model_size)
        try:
            model = self._load_model()
            result = model.transcribe(
                str(audio_path),
                word_timestamps=True,
                verbose=False,
            )
            srt_text = self._segments_to_srt(result["segments"])
            output_path.write_text(srt_text, encoding="utf-8")
            logger.info("SRT written: %s", output_path)
        except Exception as exc:
            logger.warning("Whisper failed (%s); writing empty SRT", exc)
            output_path.write_text("", encoding="utf-8")

        return output_path

    def generate_ass(self, audio_path: Path, output_path: Path) -> Path:
        """Generate Advanced SubStation Alpha subtitles (styled, karaoke-ready)."""
        srt_path = output_path.with_suffix(".srt")
        self.generate_srt(audio_path, srt_path)
        ass_text = self._srt_to_ass(srt_path.read_text(encoding="utf-8"))
        output_path.write_text(ass_text, encoding="utf-8")
        return output_path

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is None:
            import whisper  # type: ignore
            logger.info("Loading Whisper model %s …", self.model_size)
            self._model = whisper.load_model(self.model_size)
        return self._model

    @staticmethod
    def _segments_to_srt(segments: list[dict]) -> str:
        lines = []
        for i, seg in enumerate(segments, start=1):
            start = SubtitleGenerator._ts(seg["start"])
            end = SubtitleGenerator._ts(seg["end"])
            text = seg["text"].strip()
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        return "\n".join(lines)

    @staticmethod
    def _ts(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _srt_to_ass(srt_text: str) -> str:
        header = (
            "[Script Info]\n"
            "ScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial Rounded MT Bold,72,&H00FFFFFF,&H000000FF,"
            "&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,80,80,80,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )
        events = []
        import re
        blocks = re.split(r"\n\n+", srt_text.strip())
        for block in blocks:
            parts = block.strip().splitlines()
            if len(parts) < 3:
                continue
            times = parts[1].split(" --> ")
            if len(times) != 2:
                continue
            start = SubtitleGenerator._srt_ts_to_ass(times[0].strip())
            end = SubtitleGenerator._srt_ts_to_ass(times[1].strip())
            text = " ".join(parts[2:]).replace("\n", "\\N")
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
        return header + "\n".join(events) + "\n"

    @staticmethod
    def _srt_ts_to_ass(ts: str) -> str:
        # SRT: 00:00:01,500 → ASS: 0:00:01.50
        ts = ts.replace(",", ".")
        parts = ts.split(":")
        h, m, rest = parts[0], parts[1], parts[2]
        s, ms = rest.split(".")
        return f"{int(h)}:{m}:{s}.{ms[:2]}"
