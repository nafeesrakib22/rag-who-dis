#!/bin/bash
# Ollama entrypoint: start the server, pull the embedding model, then keep running.
# The model is stored in a named volume (ollama_data) so the pull only happens once.
set -e

# Start Ollama server in the background
ollama serve &
SERVE_PID=$!

# Wait until the REST API is accepting requests (uses wget, which is available in the image)
echo "[ollama-entrypoint] Waiting for Ollama to be ready..."
until wget -qO- http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done

# Pull the embedding model (no-op if already present in the volume)
echo "[ollama-entrypoint] Pulling embeddinggemma model (skipped if already cached)..."
ollama pull embeddinggemma

echo "[ollama-entrypoint] Ready. Ollama is serving embeddinggemma."

# Hand control back to the background server process
wait $SERVE_PID
