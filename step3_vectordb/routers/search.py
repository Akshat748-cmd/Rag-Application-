from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from core.searcher import get_embedded_tables, search_similar_chunks

router = APIRouter(prefix="/api", tags=["search"])

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
DB_URI = os.environ.get("DATABASE_URL", "postgresql://postgres:Akshat%402004@localhost:2004/postgres")

class SearchRequest(BaseModel):
    query: str
    source_table: str
    top_k: Optional[int] = 5
    connection_uri: Optional[str] = ""

def resolve_uri(req_uri: Optional[str]) -> str:
    return req_uri.strip() if req_uri and req_uri.strip() else DB_URI

@router.get("/tables")
async def list_tables(connection_uri: Optional[str] = ""):
    """List tables that have been embedded."""
    uri = resolve_uri(connection_uri)
    try:
        tables = get_embedded_tables(uri)
        return {"success": True, "tables": tables}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/search")
async def perform_search(req: SearchRequest):
    """Embed query and search top_k similar chunks."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    uri = resolve_uri(req.connection_uri)
    try:
        result = search_similar_chunks(uri, req.source_table, req.query, top_k=req.top_k or 5)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
