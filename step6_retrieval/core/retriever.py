"""
Step 6 — Chunks Retrieval App Core
Uses ChromaDB to retrieve the most similar text chunks based on query vector.
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import time

_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None

# Point to ChromaDB directory in Step 3
CHROMA_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
    "step3_vectordb", 
    "chroma_data"
)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print(f"[Retriever-Step6] Loading model: {_MODEL_NAME} ...")
        _embed_model = SentenceTransformer(_MODEL_NAME)
        print("[Retriever-Step6] Model loaded!")
    return _embed_model

def embed_query(query: str) -> list:
    return get_embed_model().encode(query, show_progress_bar=False, convert_to_numpy=True).tolist()

def retrieve_similar_chunks(source_table: str, query: str, top_k: int = 5) -> dict:
    t0 = time.perf_counter()
    collection_name = source_table.replace("_", "-")[:63]
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        return {
            "success": False,
            "message": f"Chroma collection '{collection_name}' not found. Please sync in Step 3 first."
        }

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    formatted_results = []
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            # Distance scoring mapping
            if distance > 1.001:
                score = 1.0 - (distance / 2.0)
            else:
                score = 1.0 - distance
            score = max(0.0, min(1.0, score))

            formatted_results.append({
                "id": results['ids'][0][i],
                "score": round(score, 4),
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })

    return {
        "success": True,
        "query": query,
        "collection": collection_name,
        "top_k": top_k,
        "elapsed_ms": elapsed_ms,
        "results": formatted_results
    }
