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

    def ask(self, question: str) -> dict:
        """
        Answer a question using retrieved context from Weaviate.
        """
        print(f"\n{'='*60}")
        print(f"QUERY: {question}")
        print('='*60)

        if self.store.count() == 0:
            return {
                "answer": "The knowledge base is empty. Please ingest a document first.",
                "sources": [],
            }

        # Step 1: Embed the question
        print("[rag] Embedding question...")
        query_vector = self.embedder.embed([question], mode="query")[0]

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
        prompt = self._build_prompt(question, retrieved_chunks)

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

    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
        """
        Build a grounded prompt for the LLM.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            label = f"[Source {i}] ({chunk['source']}, page {chunk['page']})"
            context_parts.append(f"{label}\n{chunk['text']}")

        context_block = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.
 
 Rules:
 - Base your answer SOLELY on the context below. Do not use outside knowledge.
 - Answer in the SAME LANGUAGE as the question (e.g., if the question is in Bangla, the answer must be in Bangla).
 - If the context is in a different language than the question, translate the information into the question's language.
 - Cite your sources inline using [Source N] notation.
 - If the answer cannot be found in the context, say: "I don't have enough information in the provided documents to answer this."
 
 CONTEXT:
 {context_block}
 
 QUESTION:
 {question}
 
 ANSWER:"""

        return prompt
