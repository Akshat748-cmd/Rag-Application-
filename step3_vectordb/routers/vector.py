from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from core.vectordb import get_embedded_tables, sync_to_chroma, search_chroma

router = APIRouter(prefix="/api", tags=["vector"])

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
DB_URI = os.environ.get("DATABASE_URL", "postgresql://postgres:Akshat%402004@localhost:2004/postgres")

class SyncRequest(BaseModel):
    source_table: str
    connection_uri: Optional[str] = ""

class SearchRequest(BaseModel):
    query: str
    source_table: str
    top_k: Optional[int] = 5

def resolve_uri(req_uri: Optional[str]) -> str:
    return req_uri.strip() if req_uri and req_uri.strip() else DB_URI

@router.get("/tables")
async def list_tables(connection_uri: Optional[str] = ""):
    """List tables that have been embedded in Postgres."""
    uri = resolve_uri(connection_uri)
    try:
        tables = get_embedded_tables(uri)
        return {"success": True, "tables": tables}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/sync")
async def perform_sync(req: SyncRequest):
    """Sync Postgres embeddings to ChromaDB."""
    uri = resolve_uri(req.connection_uri)
    try:
        result = sync_to_chroma(uri, req.source_table)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def perform_search(req: SearchRequest):
    """Search ChromaDB using semantic similarity."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        result = search_chroma(req.source_table, req.query, top_k=req.top_k or 5)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
