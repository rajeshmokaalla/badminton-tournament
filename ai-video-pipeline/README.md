# AI Video Generation Pipeline

Generate unique, monetisation-ready **YouTube Shorts** and **Instagram Reels** fully automatically — from a single text prompt to a polished 9:16 MP4 with narration, music, animated visuals, and burned-in subtitles.

```
User Prompt
      │
      ▼
Chat Interface (Gradio)
      │
      ▼
Llama/Qwen (Ollama) → script + image prompts + music prompt
      │
      ├── Piper TTS        → narration audio
      ├── MusicGen         → background music
      ├── SDXL / FLUX      → per-scene images
      ├── Ken Burns / SVD  → animated video clips
      └── Whisper          → SRT/ASS subtitles
      │
      ▼
FFmpeg — assemble: clips + narration + music + subtitles → final 9:16 MP4
      │
      ├── YouTube Data API  → upload as Short
      └── Meta Graph API    → upload as Reel
```

---

## Features

| Feature | Tool | Notes |
|---------|------|-------|
| Script generation | Ollama (Llama 3.2 / Qwen 2.5 / Mistral …) | Fully local, no API cost |
| Narration (TTS) | Piper | Fast, offline, many voices |
| Background music | MusicGen (`facebook/musicgen-small`) | Local inference |
| Image generation | SDXL **or** FLUX.1-schnell | Swap via `.env` |
| Video animation | Ken Burns (FFmpeg) **or** Stable Video Diffusion | SVD needs ≥24 GB VRAM |
| Subtitles | OpenAI Whisper (local) | SRT + styled ASS |
| Video assembly | FFmpeg | 9:16 vertical, H.264 |
| YouTube upload | YouTube Data API v3 | OAuth 2.0, resumable |
| Instagram upload | Meta Graph API v21 | Reels, professional account |
| UI | Gradio | Browser-based chat interface |
| Deployment | Docker + docker-compose | GPU-enabled |

---

## Quick Start

### 1 — Clone & configure

```bash
git clone https://github.com/rajeshmokaalla/badminton-tournament.git
cd badminton-tournament/ai-video-pipeline
cp .env.example .env
# Edit .env to choose your model sizes and set API keys
```

### 2 — Native installation (NVIDIA GPU recommended)

```bash
# Install system dependencies
sudo apt install ffmpeg python3.11 python3.11-venv

# Install Piper TTS binary
wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz
tar xzf piper_linux_x86_64.tar.gz -C ~/.local/bin && chmod +x ~/.local/bin/piper

# Install Ollama and pull a model
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2:3b          # or qwen2.5:7b, mistral:7b, etc.

# Create venv and install Python dependencies
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Launch the Gradio UI
python main.py
# → open http://localhost:7860
```

### 3 — Docker (easiest, GPU required)

```bash
docker compose up --build
# → open http://localhost:7860
```

### 4 — Headless CLI

```bash
python main.py --topic "The mystery of black holes" --style "dramatic and cinematic"

# With uploads
python main.py \
  --topic "5 Python tips that will change your life" \
  --upload-youtube \
  --upload-instagram \
  --ig-video-url "https://your-cdn.com/video.mp4"
```

---

## Project Structure

```
ai-video-pipeline/
├── main.py                    # Entry point (UI + CLI)
├── config.py                  # All configuration via env vars
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh
├── .env.example               # Copy to .env
├── pipeline/
│   ├── script_generator.py    # Ollama → JSON script
│   ├── narration.py           # Piper TTS
│   ├── music_generator.py     # MusicGen
│   ├── image_generator.py     # SDXL / FLUX via diffusers
│   ├── video_animator.py      # Ken Burns (FFmpeg) / SVD
│   ├── subtitle_generator.py  # Whisper → SRT/ASS
│   └── video_assembler.py     # FFmpeg final assembly
├── uploaders/
│   ├── youtube_uploader.py    # YouTube Data API v3
│   └── instagram_uploader.py  # Meta Graph API
├── ui/
│   └── gradio_app.py          # Gradio chat interface
├── utils/
│   └── helpers.py             # Shared utilities
├── outputs/                   # Generated files (git-ignored)
├── models/                    # Downloaded model weights (git-ignored)
└── credentials/               # API credentials (git-ignored)
```

---

## Hardware Requirements

| Setup | Minimum | Recommended |
|-------|---------|-------------|
| CPU-only (slow) | 16 GB RAM | 32 GB RAM |
| NVIDIA GPU (fast) | 8 GB VRAM (SDXL fp16) | 16–24 GB VRAM |
| FLUX.1 | 24 GB VRAM | 24 GB VRAM |
| SVD animation | 24 GB VRAM | 24 GB VRAM |

> **No GPU?** The pipeline falls back to CPU for all steps. Image generation will be slow (~5–10 min/image on CPU). Everything else runs comfortably on CPU.

---

## YouTube Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Create a project → enable **YouTube Data API v3**.
3. Create **OAuth 2.0 credentials** (Desktop application).
4. Download `client_secrets.json` → save to `credentials/youtube_client_secrets.json`.
5. Set `YOUTUBE_PRIVACY=private` (safe default) in `.env`.
6. On first upload, a browser window opens for OAuth consent; the token is cached for future runs.

---

## Instagram Setup

1. Create a [Meta for Developers](https://developers.facebook.com/) app.
2. Add the **Instagram Graph API** product.
3. Connect a **Professional Instagram account** (Creator or Business).
4. Generate a **long-lived access token** with `instagram_basic` + `instagram_content_publish` scopes.
5. Set `INSTAGRAM_ACCESS_TOKEN` and `INSTAGRAM_ACCOUNT_ID` in `.env`.
6. For the upload, provide a **publicly accessible HTTPS URL** to the video file (use AWS S3, Cloudflare R2, or ngrok for local testing).

---

## Configuration Reference

All options are set in `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2:3b` | LLM for script generation |
| `PIPER_MODEL` | `en_US-lessac-medium` | TTS voice |
| `MUSICGEN_MODEL` | `facebook/musicgen-small` | Music model size |
| `MUSICGEN_DURATION` | `30` | Background music length (s) |
| `IMAGE_BACKEND` | `sdxl` | `sdxl` or `flux` |
| `IMAGE_WIDTH` / `IMAGE_HEIGHT` | `1080` / `1920` | Output resolution (9:16) |
| `SVD_ENABLED` | `false` | Enable Stable Video Diffusion |
| `WHISPER_MODEL` | `base` | `tiny`/`base`/`small`/`medium`/`large` |
| `SCENE_DURATION` | `5.0` | Seconds per scene |
| `YOUTUBE_PRIVACY` | `private` | `private`/`unlisted`/`public` |

---

## Extending the Pipeline

- **Different LLM**: Change `OLLAMA_MODEL` to any model in `ollama list`. Qwen 2.5 7B produces excellent scripts.
- **Different voice**: Visit [Piper voices](https://huggingface.co/rhasspy/piper-voices) and set `PIPER_MODEL`.
- **Higher quality music**: Switch to `facebook/musicgen-medium` or `facebook/musicgen-large`.
- **Better images**: Use `IMAGE_BACKEND=flux` or mount a LoRA in `image_generator.py`.
- **Real video animation**: Set `SVD_ENABLED=true` (requires ≥24 GB VRAM).

---

## Originality & Monetisation

Each generation produces a **new, unique output** because:
- The LLM writes a fresh script with different narration and image prompts each run.
- Piper synthesises audio from the new script.
- MusicGen generates a new music track from a different prompt each time.
- SDXL/FLUX produces new images from the new prompts with different seeds.
- Whisper transcribes the new audio into new subtitles.

Platform monetisation eligibility depends on each platform's policies and review process — no tool can guarantee approval, but generating every asset from scratch maximises originality.

---

## License

MIT — see [LICENSE](LICENSE).
