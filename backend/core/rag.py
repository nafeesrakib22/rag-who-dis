"""
rag.py — The Orchestrator (Retrieval-Augmented Generation)

This module ties everything together:

  INGEST PIPELINE:
    File → loader → pages → chunker → chunks → embedder → embeddings → Weaviate

  QUERY PIPELINE (Gemini mode):
    Question → condense with history → embedder → Weaviate hybrid search
    → reranker → build full prompt (with history block) → Gemini API → answer

  QUERY PIPELINE (Local LLM mode):
    Question → embedder → Weaviate hybrid search → reranker
    → inject context into stateful KV cache session → gemma-4-E2B-it → answer
    (History lives in the model's KV cache — no re-injection needed)
"""

import logging
from collections.abc import Iterator

from . import config
from .loader import load_document
from .chunker import chunk_documents, SemanticChunker
from .embedder import Embedder
from .weaviate_store import WeaviateStore
from .reranker import Reranker
from .llm import get_llm_service

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    The full RAG pipeline: ingest documents and answer questions with citations.
    """

    def __init__(self):
        self.embedder = Embedder()
        self.store = WeaviateStore()
        self.semantic_chunker = SemanticChunker(
            self.embedder,
            breakpoint_threshold_percentile=config.SEMANTIC_PERCENTILE_THRESHOLD,
            max_chunk_size=config.CHUNK_SIZE,
        )
        self.reranker = Reranker()
        self.llm = get_llm_service()

    def ingest(self, file_path: str, source_name: str = None, progress_callback=None) -> None:
        """
        Load a document, chunk it, embed it, and store it in Weaviate.
        source_name overrides the filename stored in the DB (use when the file
        is a temp path but the original name should be preserved).
        progress_callback(stage, payload) is called at each pipeline stage so
        callers can stream progress to a client.
        """
        def _emit(stage, payload=None):
            if progress_callback:
                progress_callback(stage, payload)

        logger.info("Ingesting: %s", source_name or file_path)
        _emit("loading")
        pages = load_document(file_path, source_name=source_name)

        _emit("chunking", {"pages": len(pages)})
        chunks = chunk_documents(
            pages,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
            semantic_chunker=self.semantic_chunker
        )

        if not chunks:
            logger.warning("No chunks produced. The document may be empty.")
            return

        _emit("embedding", {"chunks": len(chunks)})
        texts = [c["text"] for c in chunks]
        logger.info("Embedding %d chunks...", len(texts))
        embeddings = self.embedder.embed(texts)

        _emit("storing", {"chunks": len(chunks)})
        self.store.add_chunks(chunks, embeddings)

        logger.info("Ingestion complete. '%s' is now searchable.", source_name or file_path)

    def ask(self, question: str, history: list[dict] = None, session_id: str = None) -> dict:
        """
        Answer a question using retrieved context from Weaviate.

        - Gemini mode: history is summarised + injected as text into the prompt.
        - Local mode:  history lives in the model's KV cache; only new context
                       and the question are sent each turn.
        """
        logger.info("Query: %s  [provider=%s]", question, config.LLM_PROVIDER)

        history = history or []

        # ── Gemini path: condense query using conversation history ─────────────
        condensed_query = question
        summary = ""
        selected_past = []
        recent_history = history[-3:]

        if config.LLM_PROVIDER == "gemini" and history:
            past_history = history[:-3]
            logger.debug("Analysing context from %d previous turns...", len(history))
            if past_history:
                summary = self._get_conversation_summary(past_history)
            initial_vector = self.embedder.embed([question], mode="query")[0]
            selected_past = self._get_relevant_past_messages(initial_vector, past_history)
            condensed_query = self._condense_query(summary, selected_past, recent_history, question)
            logger.debug("Condensed query: '%s'", condensed_query)

        # ── Local path: no query condensation (history is in KV cache) ────────
        # condensed_query remains the raw question

        if self.store.count() == 0:
            return {
                "answer": "The knowledge base is empty. Please ingest a document first.",
                "sources": [],
            }

        # ── Step 1: Embed the (possibly condensed) query ───────────────────────
        logger.debug("Embedding query: '%s'...", condensed_query)
        query_vector = self.embedder.embed([condensed_query], mode="query")[0]

        # ── Stage 1: Hybrid search ─────────────────────────────────────────────
        logger.debug("Stage 1: hybrid retrieval of top %d candidates...", config.RERANK_CANDIDATES)
        candidates = self.store.query(
            query_vector,
            n_results=config.RERANK_CANDIDATES,
            query_text=question,
        )

        if not candidates:
            return {"answer": "No relevant content found in the knowledge base.", "sources": []}

        # ── Stage 2: Re-ranking ────────────────────────────────────────────────
        if config.USE_RERANKER:
            retrieved_chunks = self.reranker.rerank(question, candidates, top_n=config.TOP_K_RESULTS)
        else:
            logger.debug("Stage 2: reranking bypassed (USE_RERANKER=False).")
            retrieved_chunks = candidates[:config.TOP_K_RESULTS]
            for c in retrieved_chunks:
                c["rerank_score"] = None

        logger.debug("Final %d chunks after re-ranking.", len(retrieved_chunks))

        # ── Step 3: Generate answer ────────────────────────────────────────────
        try:
            if config.LLM_PROVIDER == "local":
                # Local: inject context into the stateful KV cache session
                context_block = self._build_context_block(retrieved_chunks)
                answer_text = self.llm.chat(
                    question=question,
                    context_block=context_block,
                    session_id=session_id or "default",
                )
            else:
                # Gemini: build full prompt with history block
                prompt = self._build_prompt(
                    question, retrieved_chunks,
                    summary=summary,
                    selected_past=selected_past,
                    recent_history=recent_history,
                )
                answer_text = self.llm.generate_content(prompt)
        except Exception as e:
            return {"answer": f"Error calling LLM: {e}", "sources": []}

        # ── Step 4: Format source lists for both pipeline stages ───────────────
        stage1_sources = [
            {
                "n": i + 1,
                "source": c["source"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "distance": c["distance"],
                "hybrid_score": c.get("hybrid_score"),
                "text": c["text"],
                "preview": c["text"][:120] + "..." if len(c["text"]) > 120 else c["text"],
            }
            for i, c in enumerate(candidates)
        ]

        stage2_sources = [
            {
                "n": i + 1,
                "source": c["source"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "distance": c["distance"],
                "hybrid_score": c.get("hybrid_score"),
                "rerank_score": c.get("rerank_score"),
                "text": c["text"],
                "preview": c["text"][:120] + "..." if len(c["text"]) > 120 else c["text"],
            }
            for i, c in enumerate(retrieved_chunks)
        ]

        return {
            "answer": answer_text,
            "sources": stage2_sources,
            "stages": {
                "initial": stage1_sources,
                "reranked": stage2_sources,
            },
        }

    def prepare_context(
        self, question: str, history: list[dict] = None, session_id: str = None
    ) -> dict | None:
        """
        Run the full retrieval pipeline (embed → hybrid search → rerank)
        and return everything needed for generation, WITHOUT calling the LLM.

        Returns None when the knowledge base is empty.
        """
        history = history or []

        condensed_query = question
        summary = ""
        selected_past = []
        recent_history = history[-3:]

        if config.LLM_PROVIDER == "gemini" and history:
            past_history = history[:-3]
            if past_history:
                summary = self._get_conversation_summary(past_history)
            initial_vector = self.embedder.embed([question], mode="query")[0]
            selected_past = self._get_relevant_past_messages(initial_vector, past_history)
            condensed_query = self._condense_query(summary, selected_past, recent_history, question)

        if self.store.count() == 0:
            return None

        query_vector = self.embedder.embed([condensed_query], mode="query")[0]

        candidates = self.store.query(
            query_vector,
            n_results=config.RERANK_CANDIDATES,
            query_text=question,
        )
        if not candidates:
            return None

        if config.USE_RERANKER:
            retrieved_chunks = self.reranker.rerank(question, candidates, top_n=config.TOP_K_RESULTS)
        else:
            retrieved_chunks = candidates[:config.TOP_K_RESULTS]
            for c in retrieved_chunks:
                c["rerank_score"] = None

        # Build formatted source dicts
        def fmt(chunks, include_rerank=False):
            return [
                {
                    "n": i + 1,
                    "source": c["source"],
                    "page": c["page"],
                    "chunk_index": c["chunk_index"],
                    "hybrid_score": c.get("hybrid_score"),
                    **({
                        "rerank_score": c.get("rerank_score"),
                    } if include_rerank else {}),
                    "text": c["text"],
                    "preview": c["text"][:120] + "..." if len(c["text"]) > 120 else c["text"],
                }
                for i, c in enumerate(chunks)
            ]

        return {
            "question": question,
            "session_id": session_id,
            "retrieved_chunks": retrieved_chunks,
            "candidates": candidates,
            "summary": summary,
            "selected_past": selected_past,
            "recent_history": recent_history,
            "sources": fmt(retrieved_chunks, include_rerank=True),
            "stages": {
                "initial": fmt(candidates),
                "reranked": fmt(retrieved_chunks, include_rerank=True),
            },
        }

    def stream_answer(self, context: dict) -> Iterator[str]:
        """
        Given a prepared context dict from prepare_context(), stream the LLM
        response token-by-token.
        """
        chunks = context["retrieved_chunks"]
        question = context["question"]

        if config.LLM_PROVIDER == "local":
            context_block = self._build_context_block(chunks)
            # Local LLM doesn't support streaming — yield the full response
            answer = self.llm.chat(
                question=question,
                context_block=context_block,
                session_id=context.get("session_id") or "default",
            )
            yield answer
        else:
            prompt = self._build_prompt(
                question, chunks,
                summary=context.get("summary", ""),
                selected_past=context.get("selected_past", []),
                recent_history=context.get("recent_history", []),
            )
            yield from self.llm.stream_content(prompt)

    def reset_local_session(self) -> None:
        """Reset the local LLM KV cache session (no-op for Gemini mode)."""
        if config.LLM_PROVIDER == "local" and hasattr(self.llm, "reset_session"):
            self.llm.reset_session()

    # ── Prompt helpers ────────────────────────────────────────────────────────

    def _build_context_block(self, chunks: list[dict]) -> str:
        """
        Build the document context block — used by the local LLM path.
        Injected per-turn alongside the question; not stored in history.
        """
        parts = []
        for i, chunk in enumerate(chunks, 1):
            label = f"[Source {i}] ({chunk['source']}, page {chunk['page']})"
            parts.append(f"{label}\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def _build_prompt(self, question: str, chunks: list[dict], summary: str = "",
                      selected_past: list[dict] = None, recent_history: list[dict] = None) -> str:
        """
        Build a grounded prompt for the Gemini LLM.
        Includes the full history block (summary + relevant past + recent turns).
        """
        history_parts = []
        if summary:
            history_parts.append(f"SUMMARY OF PREVIOUS CONVERSATION:\n{summary}")
        if selected_past:
            history_parts.append("RELEVANT PAST STATEMENTS:")
            for m in selected_past:
                history_parts.append(f"- {m['role'].capitalize()}: {m['content']}")
        if recent_history:
            history_parts.append(
                "RECENT TURNS:\n" +
                "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent_history])
            )

        history_block = "\n\n---\n\n".join(history_parts) if history_parts else ""
        context_block = self._build_context_block(chunks)

        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context and conversation history below.
 
 Rules:
 - Base your answer SOLELY on the provided context. Do not use outside knowledge.
 - Answer in the SAME LANGUAGE as the question.
 - Cite your sources inline using [Source N] notation.
 - If the answer cannot be found in the context, say: "I don't have enough information in the provided documents to answer this."
 
{history_block}

---

CONTEXT FROM DOCUMENTS:
{context_block}
 
---
 
QUESTION:
{question}
 
ANSWER:"""

        return prompt

    # ── Gemini-only history helpers ───────────────────────────────────────────

    def _get_conversation_summary(self, past: list[dict]) -> str:
        if not past:
            return ""
        text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in past])
        prompt = (
            "Summarise the key topics, facts, and decisions discussed in these previous "
            "conversation turns for brief context reference:\n\n"
            f"{text}\n\nSUMMARY:"
        )
        try:
            return self.llm.generate_content(prompt)
        except Exception as e:
            logger.warning("Error summarising conversation history: %s", e)
            return ""

    def _get_relevant_past_messages(self, query_vector: list[float],
                                    past: list[dict], threshold=0.75) -> list[dict]:
        if not past:
            return []
        import numpy as np
        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        selected = []
        try:
            contents = [m["content"] for m in past]
            m_vectors = self.embedder.embed(contents, mode="query")
            for i, m in enumerate(past):
                if cosine(query_vector, m_vectors[i]) >= threshold:
                    selected.append(m)
        except Exception as e:
            logger.warning("Error batch-embedding past messages: %s", e)
        return selected

    def _condense_query(self, summary: str, selected: list[dict],
                        recent: list[dict], question: str) -> str:
        """
        Reformulate a follow-up question into a standalone search query.
        Only called in Gemini mode — local mode relies on the KV cache.
        """
        context = []
        if summary:
            context.append(f"CONVERSATION SUMMARY:\n{summary}")
        if selected:
            context.append("RELEVANT PAST STATEMENTS:")
            for m in selected:
                context.append(f"- {m['role']}: {m['content']}")
        context.append(
            "RECENT TURNS:\n" +
            "\n".join([f"{m['role']}: {m['content']}" for m in recent])
        )

        prompt = f"""Given the conversation context and latest question below, formulate ONE Standalone Search Query for a document database that captures exactly what information is requested. Do NOT answer the question. Only reply with the search string.

---
{chr(10).join(context)}
---

Latest follow-up question: {question}

STANDALONE SEARCH QUERY:"""
        try:
            return self.llm.generate_content(prompt)
        except Exception:
            return question
