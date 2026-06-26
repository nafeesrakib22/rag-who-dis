"""
scripts/eval.py — Offline retrieval evaluation for the RAG pipeline.

Ingests a bundled sample document (eval/seven_wonders.md), runs a fixed set of
questions against the retrieval pipeline, then reports Hit@1/3/5 and MRR.

Usage:
    python scripts/eval.py [--top-k 5] [--keep]

Flags:
    --top-k  Number of chunks to retrieve per question (default: 5)
    --keep   Skip clearing the knowledge base after the run (useful for
             inspecting results in the UI afterward)

Requires:
    - Weaviate running:  docker compose up -d
    - GOOGLE_API_KEY set in .env (only for embedding; the LLM is not called)
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ground-truth Q&A pairs
# Each "expected_keywords" list contains strings that must ALL appear (case-
# insensitive) in a retrieved chunk for it to count as a hit.  Keywords are
# chosen to be unique to one specific section of the document so that a
# retrieval failure is meaningful.
# ---------------------------------------------------------------------------

QA_PAIRS = [
    {
        "question": "How long is the Great Wall of China in kilometres?",
        "expected_keywords": ["21,196"],
    },
    {
        "question": "Who rediscovered Petra for the Western world, and in what year?",
        "expected_keywords": ["Burckhardt", "1812"],
    },
    {
        "question": "On which mountain does the Christ the Redeemer statue stand?",
        "expected_keywords": ["Corcovado"],
    },
    {
        "question": "Which Inca emperor ordered the construction of Machu Picchu?",
        "expected_keywords": ["Pachacuti"],
    },
    {
        "question": "How many total steps does El Castillo at Chichen Itza have?",
        "expected_keywords": ["365"],
    },
    {
        "question": "How many spectators could the Roman Colosseum hold?",
        "expected_keywords": ["50,000", "80,000"],
    },
    {
        "question": "Why did Shah Jahan commission the Taj Mahal?",
        "expected_keywords": ["Mumtaz Mahal"],
    },
    {
        "question": "Which of the ancient seven wonders is still standing today?",
        "expected_keywords": ["Great Pyramid", "Khufu"],
    },
    {
        "question": "Who were the Hanging Gardens of Babylon built for, and by whom?",
        "expected_keywords": ["Amytis", "Nebuchadnezzar"],
    },
    {
        "question": "What destroyed the Colossus of Rhodes?",
        "expected_keywords": ["earthquake", "226 BC"],
    },
    {
        "question": "What English word is derived from the name of the ruler buried at Halicarnassus?",
        "expected_keywords": ["mausoleum", "Mausolus"],
    },
    {
        "question": "Who built the Lighthouse of Alexandria and roughly when?",
        "expected_keywords": ["Ptolemaic"],
    },
    {
        "question": "How many votes were cast in the New Seven Wonders poll of 2007?",
        "expected_keywords": ["100 million"],
    },
    {
        "question": "What material is the Taj Mahal primarily made from?",
        "expected_keywords": ["marble"],
    },
    {
        "question": "Who burned down the Temple of Artemis at Ephesus and why?",
        "expected_keywords": ["Herostratus"],
    },
]

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _is_hit(texts: list[str], keywords: list[str]) -> bool:
    """Return True if any chunk contains ALL expected keywords (case-insensitive)."""
    for text in texts:
        lower = text.lower()
        if all(kw.lower() in lower for kw in keywords):
            return True
    return False


def hit_at_k(texts: list[str], keywords: list[str], k: int) -> bool:
    return _is_hit(texts[:k], keywords)


def reciprocal_rank(texts: list[str], keywords: list[str]) -> float:
    for rank, text in enumerate(texts, start=1):
        lower = text.lower()
        if all(kw.lower() in lower for kw in keywords):
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline RAG retrieval eval")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve per question")
    parser.add_argument("--keep", action="store_true", help="Don't clear the KB after the run")
    args = parser.parse_args()

    # Allow running from the repo root or from scripts/
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))

    from backend.core.rag import RAGPipeline
    from backend.core import config

    sample_doc = repo_root / "eval" / "seven_wonders.md"
    if not sample_doc.exists():
        print(f"[eval] ERROR: sample document not found at {sample_doc}")
        sys.exit(1)

    print("[eval] Initialising pipeline (this loads embedding + reranker models)...")
    pipeline = RAGPipeline()

    # ── Prepare a clean slate ─────────────────────────────────────────────────
    print("[eval] Clearing knowledge base...")
    pipeline.store.clear()

    print(f"[eval] Ingesting {sample_doc.name}...")
    pipeline.ingest(str(sample_doc), source_name=sample_doc.name)

    n = len(QA_PAIRS)
    top_k = args.top_k
    print(f"\n[eval] Running {n} questions  (top_k={top_k}, reranker={'on' if config.USE_RERANKER else 'off'})\n")
    print(f"  {'#':<4}  {'Hit@1':<6} {'Hit@3':<6} {'Hit@5':<6} {'RR':<6}  Question")
    print("  " + "-" * 70)

    hits = {1: 0, 3: 0, 5: 0}
    rr_scores = []

    for i, qa in enumerate(QA_PAIRS, start=1):
        question = qa["question"]
        keywords = qa["expected_keywords"]

        # Retrieve — embed then hybrid search, then optionally rerank
        query_vector = pipeline.embedder.embed([question], mode="query")[0]
        candidates = pipeline.store.query(
            query_vector,
            n_results=top_k,
            query_text=question,
        )

        if config.USE_RERANKER and candidates:
            retrieved = pipeline.reranker.rerank(question, candidates, top_n=top_k)
        else:
            retrieved = candidates

        texts = [c["text"] for c in retrieved]

        h1 = hit_at_k(texts, keywords, 1)
        h3 = hit_at_k(texts, keywords, 3)
        h5 = hit_at_k(texts, keywords, 5)
        rr = reciprocal_rank(texts, keywords)

        hits[1] += h1
        hits[3] += h3
        hits[5] += h5
        rr_scores.append(rr)

        marker = "✓" if h5 else "✗"
        print(
            f"  {marker} {i:<3}  "
            f"{'✓' if h1 else '✗':<6} "
            f"{'✓' if h3 else '✗':<6} "
            f"{'✓' if h5 else '✗':<6} "
            f"{rr:.2f}    "
            f"{question[:55]}{'...' if len(question) > 55 else ''}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    mrr = sum(rr_scores) / n
    print("\n  " + "=" * 70)
    print(f"  Results  ({n} questions, top_k={top_k})")
    print(f"  {'Hit@1':<10} {hits[1]/n:.1%}  ({hits[1]}/{n})")
    print(f"  {'Hit@3':<10} {hits[3]/n:.1%}  ({hits[3]}/{n})")
    print(f"  {'Hit@5':<10} {hits[5]/n:.1%}  ({hits[5]}/{n})")
    print(f"  {'MRR':<10} {mrr:.4f}")
    print("  " + "=" * 70)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if not args.keep:
        print("\n[eval] Clearing knowledge base...")
        pipeline.store.clear()

    pipeline.store.close()
    print("[eval] Done.")


if __name__ == "__main__":
    main()
