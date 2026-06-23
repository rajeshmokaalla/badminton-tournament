import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")          # swap to qwen2.5:7b etc.

# ── Piper TTS ─────────────────────────────────────────────────────────────────
PIPER_EXECUTABLE = os.getenv("PIPER_EXECUTABLE", "piper")
PIPER_MODEL = os.getenv("PIPER_MODEL", "en_US-lessac-medium")    # voice model name
PIPER_MODELS_DIR = Path(os.getenv("PIPER_MODELS_DIR", str(BASE_DIR / "models" / "piper")))

# ── MusicGen ──────────────────────────────────────────────────────────────────
MUSICGEN_MODEL = os.getenv("MUSICGEN_MODEL", "facebook/musicgen-small")
MUSICGEN_DURATION = int(os.getenv("MUSICGEN_DURATION", "30"))    # seconds

# ── Image generation ──────────────────────────────────────────────────────────
IMAGE_BACKEND = os.getenv("IMAGE_BACKEND", "sdxl")               # "sdxl" | "flux"
SDXL_MODEL = os.getenv("SDXL_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
FLUX_MODEL = os.getenv("FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "1080"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "1920"))            # 9:16 vertical
IMAGES_PER_SCENE = int(os.getenv("IMAGES_PER_SCENE", "1"))

# ── Stable Video Diffusion ────────────────────────────────────────────────────
SVD_ENABLED = os.getenv("SVD_ENABLED", "false").lower() == "true"
SVD_MODEL = os.getenv("SVD_MODEL", "stabilityai/stable-video-diffusion-img2vid-xt")
SVD_FRAMES = int(os.getenv("SVD_FRAMES", "25"))
SVD_FPS = int(os.getenv("SVD_FPS", "7"))

# ── Whisper ───────────────────────────────────────────────────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")               # tiny|base|small|medium|large

# ── FFmpeg ────────────────────────────────────────────────────────────────────
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "30"))
VIDEO_DURATION = int(os.getenv("VIDEO_DURATION", "60"))          # max seconds
SCENE_DURATION = float(os.getenv("SCENE_DURATION", "5.0"))       # seconds per image

# ── YouTube ───────────────────────────────────────────────────────────────────
YOUTUBE_CLIENT_SECRETS = os.getenv("YOUTUBE_CLIENT_SECRETS", str(BASE_DIR / "credentials" / "youtube_client_secrets.json"))
YOUTUBE_TOKEN_FILE = os.getenv("YOUTUBE_TOKEN_FILE", str(BASE_DIR / "credentials" / "youtube_token.json"))
YOUTUBE_CATEGORY_ID = os.getenv("YOUTUBE_CATEGORY_ID", "22")    # People & Blogs
YOUTUBE_PRIVACY = os.getenv("YOUTUBE_PRIVACY", "private")        # private|unlisted|public

# ── Instagram / Meta Graph API ────────────────────────────────────────────────
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
INSTAGRAM_ACCOUNT_ID = os.getenv("INSTAGRAM_ACCOUNT_ID", "")
META_API_VERSION = os.getenv("META_API_VERSION", "v21.0")

# ── Output paths ──────────────────────────────────────────────────────────────
SCRIPTS_DIR = OUTPUTS_DIR / "scripts"
AUDIO_DIR = OUTPUTS_DIR / "audio"
MUSIC_DIR = OUTPUTS_DIR / "music"
IMAGES_DIR = OUTPUTS_DIR / "images"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
SUBTITLES_DIR = OUTPUTS_DIR / "subtitles"

for _d in [SCRIPTS_DIR, AUDIO_DIR, MUSIC_DIR, IMAGES_DIR, VIDEOS_DIR, SUBTITLES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
