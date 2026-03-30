from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.local_llm_mlx import generate

router = APIRouter(prefix="/llm", tags=["llm-local"])

class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

@router.get("/health")
def health():
    return {"status": "ok", "engine": "mlx-local"}

@router.post("/generate")
def gen(req: GenRequest):
    text = generate(req.prompt, max_tokens=req.max_tokens)
    return {"text": text}
