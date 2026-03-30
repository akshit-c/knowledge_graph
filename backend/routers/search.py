from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.embedder import search_memory

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@router.post("/search/")
def semantic_search(payload: SearchRequest):
    results = search_memory(payload.query, payload.top_k)
    return {"results": results}
