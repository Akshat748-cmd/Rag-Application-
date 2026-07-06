"""
Step 7 — Chunks Re-ranker App Core

Note: This is a simulated hybrid reranker (dense semantic similarity + sparse keyword relevance scoring)
by default. It is not a true cross-encoder model unless USE_CROSS_ENCODER is set to true in the configurations.
If USE_CROSS_ENCODER=true, it uses sentence-transformers' CrossEncoder with 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
"""
import os
import sys
import time
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.rag_core import (
    chroma_client,
    retrieve_chunks,
    rerank_chunks
)

load_dotenv(find_dotenv())
USE_CROSS_ENCODER = os.environ.get("USE_CROSS_ENCODER", "false").lower() == "true"

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

    # 1. Retrieve more chunks initially (e.g., top_k * 2) so we have a pool to re-rank
    retrieve_count = min(top_k * 2, collection.count())
    if retrieve_count == 0:
        return {"success": True, "results": [], "elapsed_ms": 0}

    # 2. Retrieve initial chunks using shared module
    initial_chunks = retrieve_chunks(source_table, query, retrieve_count)
    
    # Add original_rank and semantic_score metadata before reranking
    for idx, chunk in enumerate(initial_chunks):
        chunk["original_rank"] = idx + 1
        chunk["semantic_score"] = chunk["score"]

    # 3. Re-rank using shared module (supports simulated hybrid and real cross-encoder)
    reranked = rerank_chunks(query, initial_chunks, top_k, use_cross_encoder=USE_CROSS_ENCODER)

    # 4. Assign new rank and calculate rank change (shift)
    final_results = []
    for rank_idx, chunk in enumerate(reranked):
        rank_change = chunk["original_rank"] - (rank_idx + 1)
        
        # If cross-encoder was used, keyword_score won't be calculated, set it to 0.0 for frontend compatibility
        keyword_score = chunk.get("keyword_score", 0.0)
        
        final_results.append({
            **chunk,
            "keyword_score": keyword_score,
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
