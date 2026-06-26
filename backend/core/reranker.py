"""
reranker.py — Cross-Encoder Re-ranking

THE PROBLEM WITH BI-ENCODERS:

Our retrieval pipeline embeds the query and each chunk independently, then
compares their vectors with cosine similarity. This is fast (O(1) lookup
against pre-computed chunk vectors), but it's imprecise.

The bi-encoder never sees the query and the chunk together — it compares
two independently-created vectors. This means it misses vocabulary gaps like:
  - Query: "output vector shape"
  - Chunk: "embedding size kept as 512"

These mean the same thing but produce vectors that aren't close enough.

HOW CROSS-ENCODERS SOLVE THIS:

A cross-encoder takes the (query, chunk) PAIR as its input — they're
concatenated and fed into the model together as a single sequence:

    Input: [CLS] What is the output shape? [SEP] The embedding size is 512. [SEP]
    Output: a single relevance score  (e.g., 8.7)

Because the model reads both texts together, it can reason about paraphrasing,
context, and semantic equivalence far better than comparing independent vectors.

WHY TWO STAGES?

Cross-encoders are slow. You can't pre-compute scores because you need the
query, which isn't known until runtime. Running a cross-encoder over ALL chunks
in a large DB would be prohibitively slow.

Solution: use both!
  Stage 1 — Bi-encoder:    fast retrieval of top-N candidates (e.g., 20)
  Stage 2 — Cross-encoder: precise re-ranking of those 20 candidates → top-K

This gives you the speed of vector search AND the precision of cross-encoders.

THE MODEL: BAAI/bge-reranker-v2-m3

A state-of-the-art multilingual reranker that supports 100+ languages including
Bangla. It significantly outperforms the previous MiniLM model on complex scripts
and cross-lingual retrieval.
"""

import logging

from sentence_transformers.cross_encoder import CrossEncoder
from . import config

logger = logging.getLogger(__name__)


class Reranker:
    """
    Wraps a CrossEncoder model to re-rank a list of retrieved chunks.

    Usage:
        reranker = Reranker()
        top_chunks = reranker.rerank(question, candidate_chunks, top_n=5)
    """

    def __init__(self, model_name: str = None):
        """
        Load the cross-encoder model.
        """
        self.model_name = model_name or config.RERANK_MODEL_NAME
        logger.info("Loading multilingual cross-encoder '%s' on CPU...", self.model_name)
        self.model = CrossEncoder(self.model_name, device="cpu")
        logger.info("Cross-encoder ready.")

    def rerank(self, query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
        """
        Score each (query, chunk) pair and return the top_n highest-scoring ones.
        """
        if not chunks:
            return []

        # Build input pairs: [(query, chunk1_text), (query, chunk2_text), ...]
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Predict returns a numpy array of float scores, one per pair.
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach the score to each chunk dict
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk_copy = dict(chunk)
            chunk_copy["rerank_score"] = round(float(score), 4)
            scored_chunks.append(chunk_copy)

        # Sort by rerank_score descending
        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.debug("Re-ranked %d candidates → kept top %d.", len(chunks), top_n)
        for i, c in enumerate(scored_chunks[:top_n], 1):
            logger.debug(
                "  [%d] score=%+.2f | %s p%s c%s (bi-enc dist=%s)",
                i, c["rerank_score"], c["source"], c["page"],
                c["chunk_index"], c.get("distance", "?"),
            )

        return scored_chunks[:top_n]
