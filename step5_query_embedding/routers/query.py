from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.embedder import embed_query_text

router = APIRouter(prefix="/api", tags=["query"])

class QueryRequest(BaseModel):
    query: str

@router.post("/embed-query")
async def perform_query_embedding(req: QueryRequest):
    """Convert search query text to vector embeddings."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    try:
        result = embed_query_text(req.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
