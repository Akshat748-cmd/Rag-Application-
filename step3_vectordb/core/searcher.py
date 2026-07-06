"""
Step 3 — Semantic Search App
Embeds user query and calculates Cosine Similarity in Python against 
embeddings stored in PostgreSQL.
"""
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Searcher] Loading model: {_MODEL_NAME} ...")
        _model = SentenceTransformer(_MODEL_NAME)
        print("[Searcher] Model loaded!")
    return _model


def embed_query(query: str) -> np.ndarray:
    """Convert query string into 384-dim numpy array."""
    model = get_model()
    # For many models, appending "query: " helps, but for all-MiniLM it's not strictly required.
    return model.encode(query, show_progress_bar=False, convert_to_numpy=True)


def get_connection(uri: str):
    return psycopg2.connect(uri)


def get_embedded_tables(uri: str) -> List[dict]:
    """Return list of tables that have embeddings in the embeddings table."""
    conn = get_connection(uri)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("""
            SELECT source_table, COUNT(*) as count 
            FROM embeddings 
            GROUP BY source_table 
            ORDER BY source_table;
        """)
        rows = cur.fetchall()
        return [{"name": r["source_table"], "count": r["count"]} for r in rows]
    except psycopg2.errors.UndefinedTable:
        # If embeddings table doesn't exist yet
        conn.rollback()
        return []
    finally:
        conn.close()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def search_similar_chunks(uri: str, source_table: str, query: str, top_k: int = 5) -> dict:
    """
    1. Embed query.
    2. Fetch all embeddings for source_table from Postgres.
    3. Calculate cosine similarity in Python (since pgvector isn't installed).
    4. Return top_k results.
    """
    query_vec = embed_query(query)
    
    conn = get_connection(uri)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT chunk_id, document_name, chunk_index, content, embedding 
        FROM embeddings 
        WHERE source_table = %s;
    """, (source_table,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return {"query": query, "results": [], "message": "Is table ka koi embedding nahi mila."}

    results = []
    for r in rows:
        chunk_vec = np.array(json.loads(r["embedding"]))
        sim = cosine_similarity(query_vec, chunk_vec)
        results.append({
            "chunk_id": r["chunk_id"],
            "document_name": r["document_name"],
            "chunk_index": r["chunk_index"],
            "content": r["content"],
            "score": round(sim, 4)
        })

    # Sort descending by score
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:top_k]

    return {
        "query": query,
        "query_embedding_preview": query_vec[:32].tolist(), # for viz
        "results": top_results
    }
