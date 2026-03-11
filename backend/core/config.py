import os
from pathlib import Path

def _load_env_file():

    current_dir = Path(__file__).parent.resolve()
    possible_paths = [
        current_dir / ".env",
        current_dir.parent / ".env",
        current_dir.parent.parent / ".env",
        Path.cwd() / ".env"
    ]
    
    env_path = None
    for p in possible_paths:
        if p.exists() and p.is_file():
            env_path = p
            break
            
    if not env_path:
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file()

# --- Gemini configuration ---
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"  
GEMINI_TEMPERATURE = 0.1

# --- Chunker configuration ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEMANTIC_PERCENTILE_THRESHOLD = 85

# --- Retrieval configuration ---
RERANK_CANDIDATES = 20
TOP_K_RESULTS = 5

# --- Weaviate configuration ---
WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.environ.get("WEAVIATE_PORT", 8080))
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", 50051))
WEAVIATE_COLLECTION_NAME = "RagChunks"
HYBRID_ALPHA = float(os.environ.get("HYBRID_ALPHA", 0.7))

# --- Models configuration ---
EMBED_MODEL_NAME = "embeddinggemma"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
