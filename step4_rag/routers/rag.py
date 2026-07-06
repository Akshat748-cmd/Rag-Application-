from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from core.generator import generate_response, chroma_client

router = APIRouter(prefix="/api", tags=["rag"])

class RAGRequest(BaseModel):
    query: str
    source_table: str
    top_k: Optional[int] = 5

@router.get("/collections")
async def list_collections():
    """List all ChromaDB collections (from Step 3 sync). No Step 3 server needed."""
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

@router.post("/generate")
async def perform_rag(req: RAGRequest):
    """Run full RAG pipeline: Retrieve + Generate."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not req.source_table.strip():
        raise HTTPException(status_code=400, detail="Source table cannot be empty.")
        
    try:
        result = generate_response(
            query=req.query,
            source_table=req.source_table,
            top_k=req.top_k
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
