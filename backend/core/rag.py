"""
rag.py — The Orchestrator (Retrieval-Augmented Generation)

This module ties everything together:

  INGEST PIPELINE:
    File → loader → pages → chunker → chunks → embedder → embeddings → ChromaDB

  QUERY PIPELINE:
    Question → embedder → query_vector → ChromaDB → top-k chunks
    → build prompt → Gemini API → answer with citations

WHY RAG AT ALL?
  LLMs are trained on data up to a cutoff date and don't know about YOUR
  documents. RAG solves this by retrieving relevant snippets from your DB
  and injecting them into the prompt as context. The LLM then generates
  an answer grounded in that context — reducing hallucinations and allowing
  it to cite sources.

THE PROMPT TEMPLATE:
  The quality of RAG depends heavily on the prompt you give the LLM.
  Our template:
    1. Tells the model to ONLY use the provided context.
    2. Provides numbered context snippets with their source labels.
    3. Asks it to cite sources using [Source N] notation.
    4. Instructs it to say "I don't know" if the answer isn't in context.
  This grounding instructions is critical — without them, LLMs will
  just answer from their training data (ignoring your documents).
"""

from . import config
from .loader import load_document
from .chunker import chunk_documents, SemanticChunker
from .embedder import Embedder
from .weaviate_store import WeaviateStore
from .reranker import Reranker
from .llm import LLMService


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
        self.llm = LLMService()

    def ingest(self, file_path: str) -> None:
        """
        Load a document, chunk it, embed it, and store it in Weaviate.
        """
        print(f"\n{'='*60}")
        print(f"INGESTING: {file_path}")
        print('='*60)

        # Step 1: Load the document into pages
        pages = load_document(file_path)

        # Step 2: Split pages into overlapping chunks
        chunks = chunk_documents(
            pages, 
            chunk_size=config.CHUNK_SIZE, 
            overlap=config.CHUNK_OVERLAP,
            semantic_chunker=self.semantic_chunker
        )

        if not chunks:
            print("[rag] No chunks produced. The document may be empty.")
            return

        # Step 3: Embed all chunk texts in one batch
        texts = [c["text"] for c in chunks]
        print(f"[rag] Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed(texts)

        # Step 4: Store chunks + embeddings in Weaviate
        self.store.add_chunks(chunks, embeddings)

        print(f"\n✅ Ingestion complete! '{file_path}' is now searchable.")

    def ask(self, question: str, history: list[dict] = None) -> dict:
        """
        Answer a question using retrieved context from Weaviate.
        """
        print(f"\n{'='*60}")
        print(f"QUERY: {question}")
        print('='*60)

        history = history or []
        recent_history = history[-3:]
        past_history = history[:-3]

        summary = ""
        selected_past = []
        condensed_query = question

        if history:
            print(f"[rag] Analyzing context from {len(history)} previous turns...")
            # 1. Summarize older turns
            if past_history:
                summary = self._get_conversation_summary(past_history)
            
            # 2. Semantic search past turns
            initial_vector = self.embedder.embed([question], mode="query")[0]
            selected_past = self._get_relevant_past_messages(initial_vector, past_history)
            
            # 3. Condense into standalone string
            condensed_query = self._condense_query(summary, selected_past, recent_history, question)
            print(f"[rag] Condensed Standalone Query: '{condensed_query}'")

        if self.store.count() == 0:
            return {
                "answer": "The knowledge base is empty. Please ingest a document first.",
                "sources": [],
            }

        # Step 1: Embed the question (condensed if available)
        print(f"[rag] Embedding query: '{condensed_query}'...")
        query_vector = self.embedder.embed([condensed_query], mode="query")[0]


        # Stage 1 — Hybrid search
        print(f"[rag] Stage 1: hybrid retrieval of top {config.RERANK_CANDIDATES} candidates...")
        candidates = self.store.query(
            query_vector,
            n_results=config.RERANK_CANDIDATES,
            query_text=question,
        )

        if not candidates:
            return {"answer": "No relevant content found in the knowledge base.", "sources": []}

        # Stage 2 — Cross-encoder re-ranking
        if config.USE_RERANKER:
            retrieved_chunks = self.reranker.rerank(question, candidates, top_n=config.TOP_K_RESULTS)
        else:
            print("[rag] Stage 2: Re-ranking bypassed (USE_RERANKER=False)")
            retrieved_chunks = candidates[:config.TOP_K_RESULTS]
            # Ensure any stale rerank scores from previous iterations or candidates are cleared
            for c in retrieved_chunks:
                c["rerank_score"] = None


        print(f"[rag] Final {len(retrieved_chunks)} chunks after re-ranking.")

        # Step 3: Build the grounded prompt
        prompt = self._build_prompt(question, retrieved_chunks, summary=summary, selected_past=selected_past, recent_history=recent_history)


        # Step 4: Generate answer via LLM Service
        try:
            answer_text = self.llm.generate_content(prompt)
        except Exception as e:
            return {
                "answer": f"Error calling LLM: {e}",
                "sources": [],
            }

        # Step 5: Format sources for both stages
        stage1_sources = [
            {
                "n": i + 1,
                "source": c["source"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "distance":     c["distance"],
                "hybrid_score": c.get("hybrid_score"),
                "text":        c["text"],
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
                "distance":     c["distance"],
                "hybrid_score": c.get("hybrid_score"),
                "rerank_score": c.get("rerank_score"),
                "text":        c["text"],
                "preview": c["text"][:120] + "..." if len(c["text"]) > 120 else c["text"],
            }
            for i, c in enumerate(retrieved_chunks)
        ]

        return {
            "answer": answer_text,
            "sources": stage2_sources, # Legacy support
            "stages": {
                "initial": stage1_sources,
                "reranked": stage2_sources
            }
        }

    def _get_conversation_summary(self, past: list[dict]) -> str:
        if not past: return ""
        text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in past])
        prompt = f"Summarise the key topics, facts, and decisions discussed in these previous conversation turns for brief context reference:\n\n{text}\n\nSUMMARY:"
        try:
            return self.llm.generate_content(prompt)
        except Exception as e:
            print(f"[rag] Error summarizing: {e}")
            return ""

    def _get_relevant_past_messages(self, query_vector: list[float], past: list[dict], threshold=0.75) -> list[dict]:
        if not past: return []
        import numpy as np
        def cosine(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        selected = []
        for m in past:
            try:
                # Embed text to compare
                m_vec = self.embedder.embed([m['content']], mode="query")[0]
                if cosine(query_vector, m_vec) >= threshold:
                    selected.append(m)
            except: pass
        return selected

    def _condense_query(self, summary: str, selected: list[dict], recent: list[dict], question: str) -> str:
        context = []
        if summary: context.append(f"CONVERSATION SUMMARY:\n{summary}")
        if selected:
            context.append("RELEVANT PAST STATEMENTS:")
            for m in selected: context.append(f"- {m['role']}: {m['content']}")
        context.append("RECENT TURNS:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in recent]))
        
        prompt = f"""Given the conversation context and latest question below, formulate ONE Standalone Search Query for a document database that captures exactly what information is requested. Do NOT answer the question. Only reply with the search string.

---
{chr(10).join(context)}
---

Latest follow-up question: {question}

STANDALONE SEARCH QUERY:"""
        try:
            return self.llm.generate_content(prompt)
        except:
            return question

    def _build_prompt(self, question: str, chunks: list[dict], summary: str = "", selected_past: list[dict] = None, recent_history: list[dict] = None) -> str:
        """
        Build a grounded prompt for the LLM.
        """
        history_parts = []
        if summary:
            history_parts.append(f"SUMMARY OF PREVIOUS CONVERSATION:\n{summary}")
        if selected_past:
            history_parts.append("RELEVANT PAST STATEMENTS:")
            for m in selected_past:
                history_parts.append(f"- {m['role'].capitalize()}: {m['content']}")
        if recent_history:
            history_parts.append("RECENT TURNS:\n" + "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent_history]))
        
        history_block = "\n\n---\n\n".join(history_parts) if history_parts else ""

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            label = f"[Source {i}] ({chunk['source']}, page {chunk['page']})"
            context_parts.append(f"{label}\n{chunk['text']}")

        context_block = "\n\n---\n\n".join(context_parts)

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

