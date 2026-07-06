"""
Chunking Router — POST /api/chunk
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.chunker import chunk_text
from core.postgres import test_db_connection, store_chunks_in_postgres
from typing import List, Dict, Optional
import asyncio
import time

router = APIRouter(prefix="/api", tags=["chunking"])


class ChunkRequest(BaseModel):
    text: str
    strategy: str = "fixed"       # fixed | recursive | sentence
    chunk_size: int = 200          # for fixed strategy
    overlap: int = 20              # overlap words
    sentences_per_chunk: int = 3   # for sentence strategy


class PostgresTestRequest(BaseModel):
    connection_uri: Optional[str] = ""
    host: Optional[str] = "localhost"
    port: Optional[int] = 2004
    database: Optional[str] = "postgres"
    user: Optional[str] = "postgres"
    password: Optional[str] = "postgres"


class PostgresStoreRequest(BaseModel):
    chunks: List[Dict]
    table_name: str = "document_chunks"
    connection_uri: Optional[str] = ""
    host: Optional[str] = "localhost"
    port: Optional[int] = 2004
    database: Optional[str] = "postgres"
    user: Optional[str] = "postgres"
    password: Optional[str] = "postgres"
    document_name: Optional[str] = "unknown"


@router.post("/chunk")
async def chunk_document(req: ChunkRequest):
    """Chunk the input text and return list of chunks."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    kwargs = {}
    if req.strategy == "fixed":
        kwargs = {"chunk_size": req.chunk_size, "overlap": req.overlap}
    elif req.strategy == "recursive":
        kwargs = {"max_size": req.chunk_size, "overlap": req.overlap}
    elif req.strategy == "sentence":
        kwargs = {"sentences_per_chunk": req.sentences_per_chunk, "overlap": req.overlap}
    elif req.strategy == "paragraph":
        kwargs = {"max_size": req.chunk_size, "overlap": req.overlap}
    elif req.strategy == "token":
        kwargs = {"chunk_size": req.chunk_size, "overlap": req.overlap}
    elif req.strategy == "sliding_window":
        kwargs = {"chunk_size": req.chunk_size, "overlap": req.overlap}

    # Run CPU-bound chunking in a thread pool to avoid blocking the async event loop
    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    chunks = await loop.run_in_executor(
        None, lambda: chunk_text(req.text, strategy=req.strategy, **kwargs)
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "success": True,
        "strategy": req.strategy,
        "total_chunks": len(chunks),
        "total_words": len(req.text.split()),
        "elapsed_ms": elapsed_ms,
        "chunks": chunks
    }


@router.post("/chunk/test-db")
async def test_postgres(req: PostgresTestRequest):
    """Test connection to PostgreSQL database."""
    success, msg = test_db_connection(
        connection_uri=req.connection_uri,
        host=req.host,
        port=req.port,
        database=req.database,
        user=req.user,
        password=req.password
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


@router.post("/chunk/store-db")
async def store_postgres(req: PostgresStoreRequest):
    """Store generated chunks into PostgreSQL database."""
    if not req.chunks:
        raise HTTPException(status_code=400, detail="No chunks provided to store")

    success, msg, count = store_chunks_in_postgres(
        chunks=req.chunks,
        table_name=req.table_name,
        connection_uri=req.connection_uri,
        host=req.host,
        port=req.port,
        database=req.database,
        user=req.user,
        password=req.password,
        document_name=req.document_name or "unknown"
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg, "inserted_count": count}


class PostgresViewRequest(BaseModel):
    table_name: str = "document_chunks"
    connection_uri: Optional[str] = ""
    host: Optional[str] = "localhost"
    port: Optional[int] = 2004
    database: Optional[str] = "postgres"
    user: Optional[str] = "postgres"
    password: Optional[str] = "postgres"
    limit: Optional[int] = 100


@router.post("/chunk/view-db")
async def view_postgres_table(req: PostgresViewRequest):
    """Fetch saved chunks from PostgreSQL table for display."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    sanitized_table = "".join(c for c in req.table_name if c.isalnum() or c == "_")
    if not sanitized_table:
        raise HTTPException(status_code=400, detail="Invalid table name")

    conn = None
    try:
        if req.connection_uri and req.connection_uri.strip():
            conn = psycopg2.connect(req.connection_uri)
        else:
            conn = psycopg2.connect(
                host=req.host, port=req.port, database=req.database,
                user=req.user, password=req.password
            )

        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Check table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = %s
            );
        """, (sanitized_table,))
        if not cur.fetchone()["exists"]:
            return {"success": True, "rows": [], "total": 0, "message": f"Table '{sanitized_table}' not found"}

        cur.execute(f"SELECT COUNT(*) as total FROM {sanitized_table};")
        total = cur.fetchone()["total"]

        cur.execute(
            f"SELECT id, chunk_index, content, word_count, strategy, created_at FROM {sanitized_table} ORDER BY chunk_index LIMIT %s;",
            (req.limit,)
        )
        rows = cur.fetchall()
        rows = [{**dict(r), "created_at": str(r["created_at"])} for r in rows]

        return {"success": True, "rows": rows, "total": total, "table": sanitized_table}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if conn:
            conn.close()
