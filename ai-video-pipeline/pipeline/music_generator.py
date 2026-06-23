"""Background music generation using Meta MusicGen (audiocraft)."""

import logging
from pathlib import Path

import scipy.io.wavfile as wavfile
import numpy as np

import config
from utils.helpers import timed

logger = logging.getLogger(__name__)


class MusicGenerator:
    def __init__(
        self,
        model_name: str = config.MUSICGEN_MODEL,
        duration: int = config.MUSICGEN_DURATION,
    ):
        self.model_name = model_name
        self.duration = duration
        self._model = None          # lazy-loaded

    # ── Public API ────────────────────────────────────────────────────────────

    @timed("music")
    def generate(self, prompt: str, output_path: Path, duration: int | None = None) -> Path:
        """Generate background music from *prompt* and save as WAV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dur = duration or self.duration

        logger.info("Generating music (%ds): %r", dur, prompt)
        try:
            model = self._load_model()
            model.set_generation_params(duration=dur)
            wav = model.generate([prompt])          # shape: (1, channels, samples)
            audio_np = wav[0].cpu().numpy()

            # MusicGen returns stereo (2, N) or mono (1, N)
            if audio_np.ndim == 2:
                audio_np = audio_np.T               # → (N, channels)

            # Normalise to int16
            audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
            sample_rate = model.sample_rate
            wavfile.write(str(output_path), sample_rate, audio_int16)

        except Exception as exc:
            logger.warning("MusicGen failed (%s); generating silence instead", exc)
            self._write_silence(output_path, dur)

        logger.info("Music written: %s", output_path)
        return output_path

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is None:
            from audiocraft.models import MusicGen  # type: ignore
            logger.info("Loading MusicGen model %s …", self.model_name)
            self._model = MusicGen.get_pretrained(self.model_name)
        return self._model

    @staticmethod
    def _write_silence(path: Path, duration_secs: int, sample_rate: int = 44100) -> None:
        samples = np.zeros((duration_secs * sample_rate,), dtype=np.int16)
        wavfile.write(str(path), sample_rate, samples)
