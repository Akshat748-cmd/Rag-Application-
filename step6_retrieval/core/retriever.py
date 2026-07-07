"""
Step 6 — Chunks Retrieval App Core
Uses ChromaDB to retrieve the most similar text chunks based on query vector.

Now supports hybrid BM25 + vector retrieval when USE_HYBRID_SEARCH=true in .env.
"""
import os
import sys
import time
from typing import List
from dotenv import load_dotenv, find_dotenv

# Add project root to sys.path so shared module is importable
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.rag_core import (
    chroma_client,
    retrieve_chunks,
    rewrite_query,
    USE_HYBRID_SEARCH,
    HAS_BM25
)

load_dotenv(find_dotenv())


def retrieve_similar_chunks(source_table: str, query: str, top_k: int = 5) -> dict:
    """Retrieve top-K similar chunks for the given query.

    Honours USE_HYBRID_SEARCH and USE_QUERY_REWRITING env flags automatically
    by delegating to shared.rag_core which handles the dispatch.
    """
    t0 = time.perf_counter()

    # Optional query rewriting — returns original query if flag is off or Gemini unavailable
    expanded_query = rewrite_query(query)

    collection_name = source_table.replace("_", "-")[:63]
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        return {
            "success": False,
            "message": f"Chroma collection '{collection_name}' not found. Please sync in Step 3 first."
        }

    # Dispatch to hybrid or pure-vector depending on flag (handled inside retrieve_chunks)
    results = retrieve_chunks(source_table, expanded_query, top_k)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    hybrid_used = USE_HYBRID_SEARCH and HAS_BM25

    return {
        "success": True,
        "query": query,
        "expanded_query": expanded_query,
        "query_rewritten": expanded_query != query,
        "collection": collection_name,
        "top_k": top_k,
        "elapsed_ms": elapsed_ms,
        "hybrid_search_used": hybrid_used,
        "results": results
    }
