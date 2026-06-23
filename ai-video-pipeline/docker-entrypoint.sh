#!/bin/bash
set -e

# Start Ollama in background and pull the configured model
echo "Starting Ollama …"
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
done
echo "Ollama ready."

MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
echo "Pulling model: $MODEL"
ollama pull "$MODEL" || echo "Warning: could not pull $MODEL (may already be cached)"

# Start the Python app
echo "Launching AI Video Pipeline …"
exec python main.py "$@"
