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
    call_gemini
)

load_dotenv(find_dotenv())
USE_CROSS_ENCODER = os.environ.get("USE_CROSS_ENCODER", "false").lower() == "true"

def generate_rag_response(source_table: str, query: str, top_k: int = 3) -> dict:
    """
    Step 8 — LLM Generation App Core.
    Retrieves, reranks (simulated hybrid or true cross-encoder), and generates response.
    """
    t0 = time.perf_counter()
    collection_name = source_table.replace("_", "-")[:63]
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        return {
            "success": False,
            "message": f"Chroma collection '{collection_name}' not found. Sync in Step 3 first."
        }

    # 1. Retrieve initial pool of chunks
    retrieve_count = min(top_k * 2, collection.count())
    if retrieve_count == 0:
        return {"success": False, "message": "No chunks found in database."}

    initial_chunks = retrieve_chunks(source_table, query, retrieve_count)
    
    # 2. Re-rank (hybrid dense+sparse or true cross-encoder)
    top_chunks = rerank_chunks(query, initial_chunks, top_k, use_cross_encoder=USE_CROSS_ENCODER)

    # 3. Assemble Prompt
    context_texts = [c["content"] for c in top_chunks]
    context = "\n\n---\n\n".join(context_texts)
    prompt = build_prompt(context, query)

    # 4. LLM Generation
    answer = call_gemini(prompt)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "success": True,
        "query": query,
        "answer": answer,
        "prompt": prompt,
        "chunks": top_chunks,
        "elapsed_ms": elapsed_ms
    }
