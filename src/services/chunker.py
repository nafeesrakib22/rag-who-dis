import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data is downloaded (silent if already exists)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


class SemanticChunker:


    def __init__(self, embedder, breakpoint_threshold_percentile: float = 95.0, max_chunk_size: int = 1000):
        """
        Args:
            embedder: An object with an .embed(list[str]) method.
            breakpoint_threshold_percentile: Percentile of distance to use as breakpoint.
            max_chunk_size: Maximum characters allowed per chunk.
        """
        self.embedder = embedder
        self.breakpoint_threshold_percentile = breakpoint_threshold_percentile
        self.max_chunk_size = max_chunk_size

    def _combine_sentences(self, sentences: list[str], buffer_size: int = 1) -> list[str]:

        combined = []
        for i in range(len(sentences)):
            start = max(0, i - buffer_size)
            end = min(len(sentences), i + buffer_size + 1)
            combined_text = " ".join(sentences[start:end])
            combined.append(combined_text)
        return combined

    def split_text(self, text: str) -> list[str]:

        if not text.strip():
            return []

        # 1. Split into sentences
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return sentences

        # 2. Create context-aware windows for better embeddings
        context_sentences = self._combine_sentences(sentences, buffer_size=1)
        
        # 3. Embed windows
        embeddings = np.array(self.embedder.embed(context_sentences))

        # 4. Calculate cosine distances between adjacent sentences
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            distance = 1 - similarity
            distances.append(distance)

        # 5. Determine breakpoint threshold based on percentile
        if not distances:
            return [" ".join(sentences)]
            
        breakpoint_threshold = np.percentile(distances, self.breakpoint_threshold_percentile)
        
        # 6. Split sentences into chunks
        chunks = []
        current_chunk_sentences = [sentences[0]]
        
        for i, distance in enumerate(distances):
            sentence = sentences[i+1]
            current_chunk_text = " ".join(current_chunk_sentences)
            
            # Check if adding the next sentence exceeds max_chunk_size
            # OR if we hit a semantic breakpoint
            if (len(current_chunk_text) + len(sentence) + 1 > self.max_chunk_size) or (distance > breakpoint_threshold):
                if current_chunk_sentences:
                    chunks.append(current_chunk_text)
                    current_chunk_sentences = []
            
            current_chunk_sentences.append(sentence)
            
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
        return chunks


def chunk_text(
    text: str,
    metadata: dict,
    chunk_size: int = 500,
    overlap: int = 50,
    semantic_chunker: SemanticChunker = None
) -> list[dict]:
    """
    Split `text` into chunks. Supports both fixed-length and semantic chunking.
    """
    if semantic_chunker:
        texts = semantic_chunker.split_text(text)
    else:
        # Fallback to fixed-length character window
        texts = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            texts.append(text[start:end].strip())
            start += chunk_size - overlap

    chunks = []
    for i, chunk_text_str in enumerate(texts):
        if chunk_text_str.strip():
            chunk = {
                "text":        chunk_text_str,
                "source":      metadata["source"],
                "page":        metadata["page"],
                "chunk_index": i,
            }
            # Pass through any extra metadata fields (e.g. price_tk, validity_days)
            for key, val in metadata.items():
                if key not in chunk:
                    chunk[key] = val
            chunks.append(chunk)
    return chunks


def chunk_documents(
    pages: list[dict], 
    chunk_size: int = 500, 
    overlap: int = 50,
    semantic_chunker: SemanticChunker = None
) -> list[dict]:
    """
    Convenience wrapper: chunk a list of page dicts.
    """
    all_chunks = []
    for page in pages:
        page_chunks = chunk_text(
            text=page["text"],
            metadata={k: v for k, v in page.items() if k != "text"},
            chunk_size=chunk_size,
            overlap=overlap,
            semantic_chunker=semantic_chunker
        )
        all_chunks.extend(page_chunks)

    method = "semantic" if semantic_chunker else "fixed-size"
    print(
        f"[chunker] {len(pages)} page(s) → {len(all_chunks)} chunks ({method})"
    )
    return all_chunks
