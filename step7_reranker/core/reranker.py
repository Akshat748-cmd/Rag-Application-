"""
Step 7 — Chunks Re-ranker App Core
Simulates a Cross-Encoder by re-scoring retrieved chunks using a hybrid
dense (semantic similarity) + sparse (keyword relevance) scoring system.
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import time
import re

_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None

CHROMA_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
    "step3_vectordb", 
    "chroma_data"
)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_MODEL_NAME)
    return _embed_model

def embed_query(query: str) -> list:
    return get_embed_model().encode(query, show_progress_bar=False, convert_to_numpy=True).tolist()

def clean_and_tokenize(text: str) -> List[str]:
    """Lowercase, remove punctuation, and split into words."""
    words = re.findall(r'\b\w+\b', text.lower())
    # Stopwords list
    stopwords = {
        "is", "a", "the", "in", "of", "to", "for", "and", "about", 
        "what", "with", "this", "or", "an", "on", "at", "by", "from"
    }
    return [w for w in words if w not in stopwords]

def calculate_keyword_score(query: str, chunk_content: str) -> float:
    """Calculate the overlap ratio of query terms in chunk content."""
    query_tokens = set(clean_and_tokenize(query))
    if not query_tokens:
        return 0.0
    
    chunk_tokens = set(clean_and_tokenize(chunk_content))
    matches = query_tokens.intersection(chunk_tokens)
    return len(matches) / len(query_tokens)

def retrieve_and_rerank(source_table: str, query: str, top_k: int = 5) -> dict:
    t0 = time.perf_counter()
    collection_name = source_table.replace("_", "-")[:63]
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        return {
            "success": False,
            "message": f"Chroma collection '{collection_name}' not found. Please sync in Step 3 first."
        }

    # 1. Retrieve more chunks initially (e.g., top_k * 2) so we have pool to re-rank
    retrieve_count = min(top_k * 2, collection.count())
    if retrieve_count == 0:
        return {"success": True, "results": [], "elapsed_ms": 0}

    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=retrieve_count
    )

    initial_chunks = []
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            if distance > 1.001:
                score = 1.0 - (distance / 2.0)
            else:
                score = 1.0 - distance
            score = max(0.0, min(1.0, score))

            initial_chunks.append({
                "id": results['ids'][0][i],
                "semantic_score": round(score, 4),
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })

    # 2. Re-rank using a hybrid scorer (60% semantic + 40% keyword overlap)
    reranked_chunks = []
    for idx, chunk in enumerate(initial_chunks):
        keyword_score = calculate_keyword_score(query, chunk["content"])
        
        # Combined score calculation
        hybrid_score = (chunk["semantic_score"] * 0.6) + (keyword_score * 0.4)
        
        reranked_chunks.append({
            **chunk,
            "original_rank": idx + 1,
            "keyword_score": round(keyword_score, 4),
            "final_score": round(hybrid_score, 4)
        })

    # Sort descending by final score
    reranked_chunks.sort(key=lambda x: x["final_score"], reverse=True)

    # Assign new rank and truncate to top_k
    final_results = []
    for rank_idx, chunk in enumerate(reranked_chunks[:top_k]):
        rank_change = chunk["original_rank"] - (rank_idx + 1)
        final_results.append({
            **chunk,
            "new_rank": rank_idx + 1,
            "rank_change": rank_change  # > 0 means moved up, < 0 means moved down, 0 means same
        })

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "success": True,
        "query": query,
        "collection": collection_name,
        "elapsed_ms": elapsed_ms,
        "results": final_results
    }
