FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# System dependencies needed by OpenCV, pdf2image, and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace models into the image layer
# Using snapshot_download (files only) instead of loading models into memory,
# which avoids the RAM spike that causes OOM crashes during docker build.
ARG HF_TOKEN
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('google/embeddinggemma-300m', token='${HF_TOKEN}'); print('embeddinggemma-300m downloaded.')"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3'); print('bge-reranker-v2-m3 downloaded.')"


# Copy the rest of the project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
