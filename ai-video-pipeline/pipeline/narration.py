"""Narration generation using Piper TTS (local, fast, offline)."""

import logging
import subprocess
import tempfile
from pathlib import Path

import config
from utils.helpers import download_piper_model, timed

logger = logging.getLogger(__name__)


class NarrationGenerator:
    def __init__(
        self,
        executable: str = config.PIPER_EXECUTABLE,
        model_name: str = config.PIPER_MODEL,
        models_dir: Path = config.PIPER_MODELS_DIR,
    ):
        self.executable = executable
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    @timed("narration")
    def generate(self, text: str, output_path: Path) -> Path:
        """Synthesise *text* to WAV at *output_path*."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_file = self._ensure_model()
        logger.info("Generating narration → %s", output_path)

        cmd = [
            self.executable,
            "--model", str(model_file),
            "--output_file", str(output_path),
        ]
        result = subprocess.run(
            cmd,
            input=text.encode(),
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Piper failed (rc={result.returncode}): {result.stderr.decode()}"
            )
        logger.info("Narration written: %s", output_path)
        return output_path

    def generate_scene_audio(self, narrations: list[str], out_dir: Path) -> list[Path]:
        """Generate one WAV per scene narration; returns ordered list of paths."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, text in enumerate(narrations):
            p = out_dir / f"scene_{i:03d}.wav"
            self.generate(text, p)
            paths.append(p)
        return paths

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_model(self) -> Path:
        """Return path to the .onnx model file, downloading if necessary."""
        onnx = self.models_dir / f"{self.model_name}.onnx"
        json_cfg = self.models_dir / f"{self.model_name}.onnx.json"
        if not onnx.exists() or not json_cfg.exists():
            logger.info("Downloading Piper model %s …", self.model_name)
            download_piper_model(self.model_name, self.models_dir)
        return onnx
