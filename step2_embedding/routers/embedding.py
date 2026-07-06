from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from core.embedder import (
    get_chunk_tables, get_stats, embed_table, get_embedded_preview
)

router = APIRouter(prefix="/api", tags=["embedding"])

DB_URI = "postgresql://postgres:Akshat%402004@localhost:2004/postgres"


class EmbedRequest(BaseModel):
    source_table: str
    connection_uri: Optional[str] = ""
    batch_size: Optional[int] = 32


class StatsRequest(BaseModel):
    source_table: str
    connection_uri: Optional[str] = ""


class PreviewRequest(BaseModel):
    source_table: str
    connection_uri: Optional[str] = ""
    limit: Optional[int] = 50


def resolve_uri(req_uri: Optional[str]) -> str:
    return req_uri.strip() if req_uri and req_uri.strip() else DB_URI


@router.get("/tables")
async def list_tables(connection_uri: Optional[str] = ""):
    """List all chunk tables available in PostgreSQL."""
    uri = resolve_uri(connection_uri)
    try:
        tables = get_chunk_tables(uri)
        return {"success": True, "tables": tables}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/stats")
async def embedding_stats(req: StatsRequest):
    """Get total / embedded / pending counts for a table."""
    uri = resolve_uri(req.connection_uri)
    try:
        stats = get_stats(uri, req.source_table)
        return {"success": True, **stats}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/embed")
async def generate_embeddings(req: EmbedRequest):
    """Generate and store embeddings for all un-embedded chunks."""
    uri = resolve_uri(req.connection_uri)
    try:
        result = embed_table(uri, req.source_table, batch_size=req.batch_size or 32)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preview")
async def preview_embeddings(req: PreviewRequest):
    """Return embedded chunks with first 32 dims for visualization."""
    uri = resolve_uri(req.connection_uri)
    try:
        rows = get_embedded_preview(uri, req.source_table, limit=req.limit or 50)
        return {"success": True, "rows": rows, "total": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
