"""
Step 3 — Vector Database (ChromaDB)
Reads pre-calculated embeddings from Postgres (Step 2) 
and inserts them into ChromaDB for Semantic Search.
"""
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional
import chromadb
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: Optional[SentenceTransformer] = None

# Initialize ChromaDB locally
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_data")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_query(query: str) -> list:
    return get_model().encode(query, show_progress_bar=False, convert_to_numpy=True).tolist()


def get_connection(uri: str):
    return psycopg2.connect(uri)


def get_embedded_tables(uri: str) -> List[dict]:
    """Fetch tables that have embeddings generated in Postgres."""
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
        conn.rollback()
        return []
    finally:
        conn.close()


def sync_to_chroma(uri: str, source_table: str) -> dict:
    """Read embeddings from Postgres and store in ChromaDB Collection."""
    # Read from Postgres
    conn = get_connection(uri)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT chunk_id, document_name, chunk_index, content, embedding 
        FROM embeddings 
        WHERE source_table = %s
    """, (source_table,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return {"success": False, "message": "No embeddings found in Postgres for this table."}

    # Connect to ChromaDB Collection
    collection_name = source_table.replace("_", "-")[:63] # ChromaDB naming rules
    collection = chroma_client.get_or_create_collection(
        name=collection_name, 
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare data for ChromaDB
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for r in rows:
        ids.append(f"chunk_{r['chunk_id']}")
        embeddings.append(json.loads(r["embedding"]))
        documents.append(r["content"])
        metadatas.append({
            "chunk_id": r["chunk_id"],
            "document_name": r["document_name"],
            "chunk_index": r["chunk_index"]
        })

    # Insert/Upsert into Chroma
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return {
        "success": True, 
        "message": f"Successfully synced {len(rows)} chunks to ChromaDB collection '{collection_name}'!",
        "count": len(rows),
        "collection": collection_name
    }


def search_chroma(source_table: str, query: str, top_k: int = 5) -> dict:
    """Use ChromaDB for Semantic Search."""
    collection_name = source_table.replace("_", "-")[:63]
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        return {"success": False, "message": "Chroma collection not found. Please Sync first."}

    query_embedding = embed_query(query)

    # Perform native Vector Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    formatted_results = []
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
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
        "results": formatted_results
    }
