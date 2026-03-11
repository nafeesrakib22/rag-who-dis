import os
import sys
from pathlib import Path

# Add current directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Load config (environment variables)
try:
    import src.services.config 
except ImportError:
    print("[error] Could not import src.services.config. Run this from the project root.")
    sys.exit(1)

from src.services.weaviate_store import WeaviateStore

def inspect_chunks():
    """
    Connect to Weaviate and print the full text of all stored chunks.
    Useful for verifying semantic chunking quality.
    """
    try:
        store = WeaviateStore()
    except Exception as e:
        print(f"\n[error] Could not connect to Weaviate: {e}")
        print("Ensure Weaviate is running (e.g., docker compose up -d)")
        return

    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*25 + "RAG CHUNK INSPECTOR" + " "*28 + "║")
    print("╚" + "═"*78 + "╝")
    
    count = store.count()
    print(f"\n[info] Found {count} chunks in database.\n")
    
    if count == 0:
        print("[!] No chunks found. Please ingest a document first.")
        store.close()
        return

    # Use iterator to stream objects from the collection
    collection = store.collection
    
    print("Loading chunks from Weaviate...\n")
    all_chunks = []
    try:
        for obj in collection.iterator():
            all_chunks.append({
                "text": obj.properties.get("text", ""),
                "source": obj.properties.get("source", "Unknown"),
                "page": obj.properties.get("page", 0),
                "chunk_idx": obj.properties.get("chunk_index", 0)
            })
    except Exception as e:
        print(f"[error] Failed to iterate collection: {e}")
        store.close()
        return
    
    # Sort: Source first, then Page, then Chunk Index
    all_chunks.sort(key=lambda x: (x["source"], x["page"], x["chunk_idx"]))
    
    last_source = None
    for i, c in enumerate(all_chunks, 1):
        if c["source"] != last_source:
            print(f"\n{'='*80}")
            print(f"📄 DOCUMENT: {c['source']}")
            print(f"{'='*80}")
            last_source = c["source"]
            
        print(f"\n--- [Chunk {i}] Page {c['page']} | Index {c['chunk_idx']} ---")
        print(c["text"])
        print("-" * 40)
        
    print(f"\n[done] Finished printing {len(all_chunks)} chunks.")
    store.close()

if __name__ == "__main__":
    try:
        inspect_chunks()
    except KeyboardInterrupt:
        print("\n[!] Inspection cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[error] Unexpected error: {e}")
        sys.exit(1)
