import os
import sys
from typing import List, Optional

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.rag_core import (
    chroma_client,
    retrieve_chunks,
    build_prompt,
    call_gemini,
    rewrite_query
)


def generate_response(query: str, source_table: str, top_k: int = 5) -> dict:
    """
    Step 4 — Generates LLM response by retrieving chunks using shared core RAG module.

    If USE_QUERY_REWRITING=true in .env, Gemini first rewrites the query for better
    retrieval recall. Both the original and expanded queries are returned for transparency.
    """
    # 1. Optional query rewriting (no-op if flag is off or Gemini unavailable)
    expanded_query = rewrite_query(query)

    try:
        # 2. Retrieve similar chunks using cosine space (or BM25+vector if flag on)
        chunks = retrieve_chunks(source_table, expanded_query, top_k)
    except Exception as e:
        return {"success": False, "message": str(e)}

    if not chunks:
        return {"success": False, "message": "No relevant context found in Vector DB."}

    # 3. Build prompt and generate response using shared LLM module
    context_texts = [c["content"] for c in chunks]
    context = "\n\n---\n\n".join(context_texts)
    prompt = build_prompt(context, expanded_query)

    answer = call_gemini(prompt)

    return {
        "success": True,
        "query": query,
        "expanded_query": expanded_query,
        "query_rewritten": expanded_query != query,
        "answer": answer,
        "retrieved_chunks": chunks
    }
