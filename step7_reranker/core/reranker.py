"""
Step 7 — Chunks Re-ranker App Core

Default: simulated hybrid reranker (dense semantic + sparse keyword scoring).
USE_CROSS_ENCODER=true  → true CrossEncoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
USE_HYBRID_SEARCH=true  → BM25 + vector hybrid retrieval before re-ranking
USE_QUERY_REWRITING=true → Gemini query expansion before retrieval
"""
import os
import sys
import time
from typing import List
from dotenv import load_dotenv, find_dotenv

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.rag_core import (
    chroma_client,
    retrieve_chunks,
    rerank_chunks,
    rewrite_query,
    USE_HYBRID_SEARCH,
    HAS_BM25
)

load_dotenv(find_dotenv())
USE_CROSS_ENCODER = os.environ.get("USE_CROSS_ENCODER", "false").lower() == "true"


def retrieve_and_rerank(source_table: str, query: str, top_k: int = 5) -> dict:
    """Retrieve chunks then re-rank them.

    Pipeline:
      1. Optional query rewriting (USE_QUERY_REWRITING flag)
      2. Retrieve initial pool (pure-vector OR BM25+vector per USE_HYBRID_SEARCH flag)
      3. Re-rank (simulated hybrid OR true CrossEncoder per USE_CROSS_ENCODER flag)
    """
    t0 = time.perf_counter()

    # Step 1 — Query rewriting (no-op if flag is off)
    expanded_query = rewrite_query(query)

    collection_name = source_table.replace("_", "-")[:63]
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        return {
            "success": False,
            "message": f"Chroma collection '{collection_name}' not found. Please sync in Step 3 first."
        }

    # Step 2 — Retrieve initial pool (hybrid or pure-vector via shared dispatch)
    retrieve_count = min(top_k * 2, collection.count())
    if retrieve_count == 0:
        return {"success": True, "results": [], "elapsed_ms": 0}

    initial_chunks = retrieve_chunks(source_table, expanded_query, retrieve_count)

    # Add original_rank and semantic_score metadata before reranking
    for idx, chunk in enumerate(initial_chunks):
        chunk["original_rank"] = idx + 1
        chunk["semantic_score"] = chunk["score"]

    # Step 3 — Re-rank (supports simulated hybrid and real cross-encoder)
    reranked = rerank_chunks(expanded_query, initial_chunks, top_k, use_cross_encoder=USE_CROSS_ENCODER)

    # Step 4 — Assign new rank and calculate rank change (shift)
    final_results = []
    for rank_idx, chunk in enumerate(reranked):
        rank_change = chunk["original_rank"] - (rank_idx + 1)
        # If cross-encoder was used, keyword_score won't be calculated; set 0.0 for frontend
        keyword_score = chunk.get("keyword_score", 0.0)
        final_results.append({
            **chunk,
            "keyword_score": keyword_score,
            "new_rank": rank_idx + 1,
            "rank_change": rank_change   # >0 = moved up, <0 = moved down, 0 = same
        })

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    hybrid_used = USE_HYBRID_SEARCH and HAS_BM25

    return {
        "success": True,
        "query": query,
        "expanded_query": expanded_query,
        "query_rewritten": expanded_query != query,
        "collection": collection_name,
        "elapsed_ms": elapsed_ms,
        "hybrid_search_used": hybrid_used,
        "results": final_results
    }
