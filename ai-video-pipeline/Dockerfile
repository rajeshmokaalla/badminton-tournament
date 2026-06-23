FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ── System dependencies ───────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg \
        libsndfile1 \
        wget curl git \
        fonts-urw-base35 \
    && rm -rf /var/lib/apt/lists/*

# ── Python alias ─────────────────────────────────────────────────────────────
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

# ── Piper TTS ─────────────────────────────────────────────────────────────────
ARG PIPER_VERSION=2023.11.14-2
RUN wget -q "https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_x86_64.tar.gz" \
        -O /tmp/piper.tar.gz \
    && tar -xzf /tmp/piper.tar.gz -C /usr/local/bin \
    && rm /tmp/piper.tar.gz \
    && chmod +x /usr/local/bin/piper

# ── Ollama ────────────────────────────────────────────────────────────────────
RUN curl -fsSL https://ollama.ai/install.sh | sh

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# ── App code ──────────────────────────────────────────────────────────────────
COPY . .

# ── Create output directories ─────────────────────────────────────────────────
RUN mkdir -p outputs/{scripts,audio,music,images,videos,subtitles} \
             models/piper credentials

# ── Ports ─────────────────────────────────────────────────────────────────────
EXPOSE 7860
# Gradio UI
EXPOSE 11434
# Ollama API

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "7860"]
