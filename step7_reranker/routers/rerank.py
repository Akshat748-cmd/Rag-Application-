from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from core.reranker import retrieve_and_rerank, chroma_client

router = APIRouter(prefix="/api", tags=["rerank"])

class RerankRequest(BaseModel):
    query: str
    source_table: str
    top_k: Optional[int] = 5

@router.get("/collections")
async def list_collections():
    """List available ChromaDB collections."""
    try:
        cols = chroma_client.list_collections()
        result = []
        for c in cols:
            col = chroma_client.get_collection(c.name)
            result.append({
                "name": c.name,
                "count": col.count()
            })
        return {"success": True, "collections": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rerank")
async def perform_reranking(req: RerankRequest):
    """Retrieve similar chunks and apply re-ranking scoring."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not req.source_table.strip():
        raise HTTPException(status_code=400, detail="Source table cannot be empty.")
    try:
        result = retrieve_and_rerank(
            source_table=req.source_table,
            query=req.query,
            top_k=req.top_k or 5
        )
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
