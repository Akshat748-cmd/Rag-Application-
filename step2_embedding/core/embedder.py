"""
Step 2 — Embedding App
Generates vector embeddings for chunks stored in PostgreSQL
using sentence-transformers (all-MiniLM-L6-v2, 384 dimensions).
"""
import json
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from typing import List, Optional
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {_MODEL_NAME} ...")
        _model = SentenceTransformer(_MODEL_NAME)
        print(f"[Embedder] Model loaded! Dimensions: 384")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Convert list of texts -> list of 384-dim vectors."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def get_connection(uri: str):
    return psycopg2.connect(uri)


def ensure_embeddings_table(conn):
    """Create the embeddings table if it doesn't exist."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id            SERIAL PRIMARY KEY,
            source_table  VARCHAR(255) NOT NULL,
            chunk_id      INTEGER      NOT NULL,
            document_name VARCHAR(255) DEFAULT 'unknown',
            chunk_index   INTEGER      DEFAULT 0,
            content       TEXT         NOT NULL,
            embedding     TEXT         NOT NULL,
            model_name    VARCHAR(100) DEFAULT 'all-MiniLM-L6-v2',
            dimensions    INTEGER      DEFAULT 384,
            created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_table, chunk_id)
        );
    """)
    conn.commit()
    cur.close()


def get_chunk_tables(uri: str) -> List[dict]:
    """Return all tables that have a 'content' column (chunk tables).
    Excludes system/non-chunk tables like embeddings, address, customer etc.
    """
    # Tables to always exclude
    EXCLUDE_TABLES = {"embeddings", "address", "customer", "customer_addresses", "customers"}

    conn = get_connection(uri)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT t.tablename,
               (SELECT COUNT(*) FROM information_schema.columns c2
                WHERE c2.table_name = t.tablename AND c2.column_name = 'content') > 0 AS has_content
        FROM pg_tables t
        WHERE t.schemaname = 'public'
        ORDER BY t.tablename;
    """)
    rows = cur.fetchall()
    conn.close()
    return [
        {"name": r["tablename"]}
        for r in rows
        if r["has_content"] and r["tablename"] not in EXCLUDE_TABLES
    ]


def get_stats(uri: str, source_table: str) -> dict:
    """Chunks total vs embedded count for a given table."""
    conn = get_connection(uri)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    sanitized = "".join(c for c in source_table if c.isalnum() or c == "_")

    cur.execute(f"SELECT COUNT(*) AS total FROM {sanitized};")
    total = cur.fetchone()["total"]

    ensure_embeddings_table(conn)
    cur.execute("SELECT COUNT(*) AS embedded FROM embeddings WHERE source_table = %s;",
                (source_table,))
    embedded = cur.fetchone()["embedded"]

    conn.close()
    return {"total": total, "embedded": embedded, "pending": total - embedded}


def embed_table(uri: str, source_table: str, batch_size: int = 32) -> dict:
    """
    Embed all un-embedded chunks from source_table.
    Stores results in the `embeddings` table.
    """
    sanitized = "".join(c for c in source_table if c.isalnum() or c == "_")
    conn = get_connection(uri)
    ensure_embeddings_table(conn)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute(f"""
        SELECT c.id,
               c.content,
               COALESCE(c.chunk_index, 0)       AS chunk_index,
               COALESCE(c.document_name, 'unknown') AS document_name
        FROM {sanitized} c
        LEFT JOIN embeddings e
               ON e.source_table = %s AND e.chunk_id = c.id
        WHERE e.id IS NULL
        ORDER BY c.id;
    """, (source_table,))
    rows = cur.fetchall()

    if not rows:
        conn.close()
        return {"inserted": 0, "message": "Sab chunks pehle se embed hain!"}

    inserted = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i: i + batch_size]
        texts = [r["content"] for r in batch]
        vecs  = embed_texts(texts)

        values = [
            (source_table, r["id"], r["document_name"], r["chunk_index"],
             r["content"], json.dumps(v), _MODEL_NAME, len(v))
            for r, v in zip(batch, vecs)
        ]
        execute_values(cur, """
            INSERT INTO embeddings
              (source_table, chunk_id, document_name, chunk_index,
               content, embedding, model_name, dimensions)
            VALUES %s
            ON CONFLICT (source_table, chunk_id) DO NOTHING;
        """, values)
        conn.commit()
        inserted += len(batch)

    conn.close()
    return {
        "inserted": inserted,
        "model": _MODEL_NAME,
        "dimensions": 384,
        "message": f"{inserted} chunks embed ho gaye! Model: {_MODEL_NAME}, Dims: 384"
    }


def get_embedded_preview(uri: str, source_table: str, limit: int = 50) -> List[dict]:
    """Fetch embedded chunks with first 32 dims for bar chart visualization."""
    conn = get_connection(uri)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT id, chunk_id, document_name, chunk_index,
               content, embedding, model_name, dimensions, created_at
        FROM embeddings
        WHERE source_table = %s
        ORDER BY chunk_index
        LIMIT %s;
    """, (source_table, limit))
    rows = cur.fetchall()
    conn.close()

    result = []
    for r in rows:
        vec = json.loads(r["embedding"])
        result.append({
            "id":              r["id"],
            "chunk_id":        r["chunk_id"],
            "document_name":   r["document_name"],
            "chunk_index":     r["chunk_index"],
            "content_preview": r["content"][:120],
            "content":         r["content"],
            "embedding_preview": vec[:32],
            "dimensions":      r["dimensions"],
            "model_name":      r["model_name"],
            "created_at":      str(r["created_at"]),
        })
    return result
