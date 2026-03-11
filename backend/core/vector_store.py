"""
vector_store.py — ChromaDB Vector Store

WHAT IS CHROMADB?

ChromaDB is a vector database: a database optimized for storing and querying
high-dimensional vectors (embeddings). Unlike a regular database that finds
rows by exact ID or value, a vector DB finds rows by SIMILARITY.

When you query it with an embedding vector, ChromaDB computes the cosine
similarity between your query vector and all stored vectors, then returns
the top-k most similar ones. This is the heart of the "Retrieval" in RAG.

HOW CHROMADB STORES DATA:

ChromaDB organizes data into "collections" (like tables). Each item in a
collection has three parts:
  1. id         — a unique string identifier
  2. embedding  — the vector (list of floats)
  3. document   — the raw text (optional but useful)
  4. metadata   — a dict of extra info (source, page, chunk_index)

We use PersistentClient so data is saved to disk between runs.
Without persistence, everything would be lost when the Python process exits.

COSINE SIMILARITY (the distance metric):

ChromaDB defaults to L2 (Euclidean) distance. We switch to cosine similarity
because sentence-transformer embeddings are trained with cosine similarity in
mind. Two vectors pointing in the same direction = similar meaning, regardless
of their magnitudes.
"""

import chromadb
from chromadb.config import Settings


CHROMA_DB_PATH = "chroma_db"   # relative to where you run the script
COLLECTION_NAME = "rag_chunks"


class ChromaStore:
    """
    A thin wrapper around ChromaDB that makes it easy to add chunks and query them.

    Usage:
        store = ChromaStore()
        store.add_chunks(chunks, embeddings)
        results = store.query(query_embedding, n_results=5)
    """

    def __init__(self, db_path: str = CHROMA_DB_PATH):
        """
        Initialize a PersistentClient pointing at `db_path`.

        PersistentClient saves data to disk so your indexed documents
        survive between Python sessions — no need to re-ingest every time.

        Args:
            db_path: Directory where ChromaDB will save its SQLite + vector files.
        """
        print(f"[vector_store] Connecting to ChromaDB at '{db_path}'...")
        self.client = chromadb.PersistentClient(path=db_path)

        # get_or_create_collection: if collection already exists, open it;
        # otherwise create it fresh. This means we can safely call this
        # every time without worrying about duplicate collections.
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )
        print(f"[vector_store] Collection '{COLLECTION_NAME}' ready. "
              f"Current item count: {self.collection.count()}")

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """
        Insert chunks + their embeddings into ChromaDB.

        ChromaDB requires that every item has a unique string ID.
        We derive it from source + page + chunk_index so re-ingesting
        the same file replaces old entries (upsert behaviour).

        Args:
            chunks:     List of chunk dicts (output of chunker.py).
            embeddings: Parallel list of embedding vectors (same order as chunks).
        """
        if not chunks:
            print("[vector_store] No chunks to add.")
            return

        ids = []
        documents = []
        metadatas = []

        for chunk, embedding in zip(chunks, embeddings):
            # Build a stable, unique ID from the chunk's origin
            chunk_id = f"{chunk['source']}__p{chunk['page']}__c{chunk['chunk_index']}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
            })

        # upsert = insert OR update if ID already exists
        # This prevents duplicates if you ingest the same file twice
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        print(f"[vector_store] Upserted {len(chunks)} chunks. "
              f"Total in DB: {self.collection.count()}")

    def query(self, query_embedding: list[float], n_results: int = 5) -> list[dict]:
        """
        Find the top-k chunks most similar to `query_embedding`.

        ChromaDB computes cosine similarity between `query_embedding` and
        every stored vector, then returns the `n_results` closest ones.

        Args:
            query_embedding: The embedding of the user's question.
            n_results:       How many chunks to retrieve.

        Returns:
            List of result dicts, sorted by similarity (best first):
            [
                {
                    "text": "...",
                    "source": "myfile.pdf",
                    "page": 2,
                    "chunk_index": 4,
                    "distance": 0.12,   ← lower = more similar (cosine distance)
                },
                ...
            ]
        """
        n_results = min(n_results, self.collection.count())
        if n_results == 0:
            print("[vector_store] Collection is empty. Ingest a document first.")
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns lists-of-lists (one list per query).
        # Since we only sent one query, we take index [0] of each.
        formatted = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            formatted.append({
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "chunk_index": meta["chunk_index"],
                "distance": round(dist, 4),
            })

        return formatted

    def count(self) -> int:
        """Return how many chunks are currently stored."""
        return self.collection.count()
