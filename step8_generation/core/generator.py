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
    rerank_chunks,
    build_prompt,
    call_gemini,
    rewrite_query,
    USE_HYBRID_SEARCH,
    HAS_BM25
)

load_dotenv(find_dotenv())
USE_CROSS_ENCODER = os.environ.get("USE_CROSS_ENCODER", "false").lower() == "true"


def generate_rag_response(source_table: str, query: str, top_k: int = 3) -> dict:
    """
    Step 8 — LLM Generation App Core.

    Pipeline:
      1. Optional query rewriting via Gemini (USE_QUERY_REWRITING flag)
      2. Retrieve initial pool (pure-vector OR BM25+vector per USE_HYBRID_SEARCH flag)
      3. Re-rank (simulated hybrid OR true CrossEncoder per USE_CROSS_ENCODER flag)
      4. Build prompt + generate answer with Gemini

    Returns original_query, expanded_query, and hybrid_search_used for UI transparency.
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
            "message": f"Chroma collection '{collection_name}' not found. Sync in Step 3 first."
        }

    # Step 2 — Retrieve initial pool
    retrieve_count = min(top_k * 2, collection.count())
    if retrieve_count == 0:
        return {"success": False, "message": "No chunks found in database."}

    initial_chunks = retrieve_chunks(source_table, expanded_query, retrieve_count)

    # Step 3 — Re-rank (hybrid dense+sparse or true cross-encoder)
    top_chunks = rerank_chunks(expanded_query, initial_chunks, top_k, use_cross_encoder=USE_CROSS_ENCODER)

    # Step 4 — Assemble Prompt + Generate
    context_texts = [c["content"] for c in top_chunks]
    context = "\n\n---\n\n".join(context_texts)
    prompt = build_prompt(context, expanded_query)
    answer = call_gemini(prompt)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    hybrid_used = USE_HYBRID_SEARCH and HAS_BM25

    return {
        "success": True,
        "query": query,
        "expanded_query": expanded_query,
        "query_rewritten": expanded_query != query,
        "answer": answer,
        "prompt": prompt,
        "chunks": top_chunks,
        "elapsed_ms": elapsed_ms,
        "hybrid_search_used": hybrid_used
    }
