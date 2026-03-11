

import requests
from . import config


class Embedder:

    def __init__(self, ollama_url: str = None, model: str = None):

        self.ollama_url = ollama_url or config.OLLAMA_URL
        self.model = model or config.EMBED_MODEL_NAME

        print(f"[embedder] Connecting to Ollama for '{self.model}' embeddings...")
        try:
            test = self._call_ollama(["startup dimension check"])
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.ollama_url}.\n"
                "Make sure Ollama is running: ollama serve\n"
                f"Then pull the model:        ollama pull {self.model}"
            )

        self.dimension = len(test[0])
        print(f"[embedder] Ready. Vector dimension: {self.dimension}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    BATCH_SIZE = 32  # texts per Ollama request — lower if you get timeouts

    def embed(self, texts: list[str], mode: str = "document") -> list[list[float]]:
        """
        Embed texts in small batches to avoid Ollama timeouts on large documents.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self.BATCH_SIZE):
            batch = texts[start : start + self.BATCH_SIZE]
            end = min(start + self.BATCH_SIZE, total)
            print(f"[embedder] Embedding {end}/{total}...", end="\r", flush=True)
            all_embeddings.extend(self._call_ollama(batch))

        if total > self.BATCH_SIZE:
            print()  # newline after progress line
        return all_embeddings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_ollama(self, texts: list[str]) -> list[list[float]]:

        resp = requests.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=300,  # 5 min per batch
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]
