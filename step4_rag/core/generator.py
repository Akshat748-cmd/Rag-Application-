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
    call_gemini
)

def generate_response(query: str, source_table: str, top_k: int = 5) -> dict:
    """
    Step 4 — Generates LLM response by retrieving chunks using shared core RAG module.
    """
    try:
        # 1. Retrieve similar chunks using cosine space
        chunks = retrieve_chunks(source_table, query, top_k)
    except Exception as e:
        return {"success": False, "message": str(e)}
        
    if not chunks:
        return {"success": False, "message": "No relevant context found in Vector DB."}
        
    # 2. Build prompt and generate response using shared LLM module
    context_texts = [c["content"] for c in chunks]
    context = "\n\n---\n\n".join(context_texts)
    prompt = build_prompt(context, query)
    
    answer = call_gemini(prompt)
    
    return {
        "success": True,
        "query": query,
        "answer": answer,
        "retrieved_chunks": chunks
    }
