"""Shared utility functions."""

import functools
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_PIPER_RELEASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"


def timed(label: str):
    """Decorator that logs elapsed time for a pipeline step."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            logger.info("[%s] %.1fs", label.upper(), elapsed)
            return result
        return wrapper
    return decorator


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def download_piper_model(model_name: str, models_dir: Path) -> None:
    """
    Download a Piper voice model (.onnx + .onnx.json) from HuggingFace.

    Model naming convention: <lang>_<region>-<name>-<quality>
    e.g. en_US-lessac-medium
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    parts = model_name.split("-")
    if len(parts) < 3:
        raise ValueError(f"Unexpected Piper model name format: {model_name!r}")

    lang_region = parts[0]           # e.g. en_US
    lang = lang_region.split("_")[0] # e.g. en
    subpath = f"{lang}/{lang_region}/{model_name}"

    for ext in [".onnx", ".onnx.json"]:
        url = f"{_PIPER_RELEASE}/{subpath}{ext}"
        dest = models_dir / f"{model_name}{ext}"
        if dest.exists():
            continue
        logger.info("Downloading %s …", url)
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        logger.info("Saved %s", dest)


def slug(text: str, max_len: int = 50) -> str:
    """Convert *text* to a filesystem-safe slug."""
    import re
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return s[:max_len]


def ensure_ffmpeg() -> None:
    """Raise an informative error if ffmpeg is not on PATH."""
    import shutil
    if not shutil.which("ffmpeg"):
        raise EnvironmentError(
            "ffmpeg not found. Install it:\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  macOS:         brew install ffmpeg\n"
            "  Docker:        see Dockerfile"
        )
