"""
embedder.py — Local HuggingFace embedding via sentence-transformers

Loads google/embeddinggemma-300m in-process (no Ollama required).
The model is downloaded to the HuggingFace cache on first use
(~600 MB) and reused from disk on subsequent starts.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from . import config


class Embedder:

    BATCH_SIZE = 32

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBED_MODEL_NAME

        print(f"[embedder] Loading '{self.model_name}' from HuggingFace (in-process)...")
        token = config.HF_TOKEN or None
        self.model = SentenceTransformer(self.model_name, token=token, device="cpu")


        # Detect dimension with a startup test
        test = self.model.encode(["startup dimension check"], convert_to_numpy=True)
        self.dimension = test.shape[1]
        print(f"[embedder] Ready. Vector dimension: {self.dimension}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str], mode: str = "document") -> list[list[float]]:
        """
        Embed texts in batches and return a list of float vectors.
        The `mode` argument is kept for API compatibility but is unused
        (sentence-transformers handles query/document symmetry internally).
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self.BATCH_SIZE):
            batch = texts[start : start + self.BATCH_SIZE]
            end = min(start + self.BATCH_SIZE, total)
            print(f"[embedder] Embedding {end}/{total}...", end="\r", flush=True)
            vecs = self.model.encode(batch, convert_to_numpy=True)
            all_embeddings.extend(vecs.tolist())

        if total > self.BATCH_SIZE:
            print()  # newline after progress line
        return all_embeddings
