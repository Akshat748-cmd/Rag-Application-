"""
Step 5 — Query Embeddings App Core
Embeds the user's search query into a 384-dimensional vector.
"""
from sentence_transformers import SentenceTransformer
from typing import Optional, List
import time

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: Optional[SentenceTransformer] = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embedder-Step5] Loading model: {_MODEL_NAME} ...")
        _model = SentenceTransformer(_MODEL_NAME)
        print("[Embedder-Step5] Model loaded!")
    return _model

def embed_query_text(query: str) -> dict:
    """Embed query text and measure time taken."""
    t0 = time.perf_counter()
    model = get_model()
    embedding = model.encode(query, show_progress_bar=False, convert_to_numpy=True)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    
    embedding_list = embedding.tolist()
    return {
        "success": True,
        "query": query,
        "embedding": embedding_list,
        "dimensions": len(embedding_list),
        "model_name": _MODEL_NAME,
        "elapsed_ms": elapsed_ms
    }
