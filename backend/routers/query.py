from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.memory_search import search
from backend.services.local_llm_mlx import generate
import traceback
from typing import Literal
import sqlite3
from pathlib import Path

DB_PATH = Path("backend/data/memory.sqlite")

def fetch_doc_summaries(doc_ids: list[str]) -> dict[str, str]:
    if not doc_ids:
        return {}
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    placeholders = ",".join(["?"] * len(doc_ids))
    rows = cur.execute(
        f"SELECT doc_id, summary FROM documents WHERE doc_id IN ({placeholders})",
        doc_ids
    ).fetchall()

    con.close()
    return {doc_id: (summary or "").strip() for doc_id, summary in rows}



router = APIRouter(prefix="/query", tags=["query"])

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    mode: Literal["full", "summary"] = "full"

NOT_IN_MEMORY = "NOT_IN_MEMORY"

@router.post("")
def query_memory(req: QueryRequest):
    try:
        hits = search(req.question, req.top_k)

        # Hard gating rule: if retrieval is weak, refuse.
        # (Tune later, but this is paper-friendly and prevents hallucinations.)
        if not hits:
            return {"answer": NOT_IN_MEMORY, "sources": [], "trace": {"reason": "no_hits", "mode_used": req.mode}}

        # Optional: require at least one strong similarity
        max_score = max(h["score"] for h in hits)
        if max_score < 0.25:  # tune as needed (0.2–0.35 typical)
            return {"answer": NOT_IN_MEMORY, "sources": hits, "trace": {"reason": "weak_retrieval", "max_score": max_score, "mode_used": req.mode}}

        # Collect doc_ids from retrieved chunks
        doc_ids = []
        for r in hits:
            did = r.get("doc_id")
            if did and did not in doc_ids:
                doc_ids.append(did)

        summaries = fetch_doc_summaries(doc_ids)

        if req.mode == "summary":
            # Build context from document summaries (fallback to chunk text if missing)
            context_parts = []
            for did in doc_ids:
                s = summaries.get(did, "")
                if s:
                    context_parts.append(f"[DOC {did} SUMMARY]\n{s}")
            # fallback if no summaries exist
            if not context_parts:
                for r in hits:
                    context_parts.append(f"[CHUNK]\n{r.get('text','')}")
            context = "\n\n".join(context_parts)
        else:
            # full mode: use chunks as context (your current behavior)
            context_parts = []
            for r in hits:
                context_parts.append(f"[CHUNK score={r.get('score')} doc={r.get('doc_id')} idx={r.get('chunk_index')}]\n{r.get('text','')}")
            context = "\n\n".join(context_parts)

        prompt = f"""Answer using only the provided context.
If the answer is not present in the context, output exactly: {NOT_IN_MEMORY}.

CONTEXT:
{context}

QUESTION:
{req.question}
"""

        answer = generate(prompt)  # this calls your Phi model via MLX/Ollama wrapper
        answer = answer.strip()
        
        # Normalize NOT_IN_MEMORY responses
        if answer.upper().strip() == NOT_IN_MEMORY:
            answer = NOT_IN_MEMORY

        max_score = max(h["score"] for h in hits) if hits else 0.0
        return {"answer": answer, "sources": hits, "trace": {"mode_used": req.mode, "max_score": max_score}}
    
    except FileNotFoundError as e:
        print(f"FileNotFoundError in query: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail=f"Search index not available: {str(e)}"
        )
    except RuntimeError as e:
        print(f"RuntimeError in query (likely MLX generation failed): {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )
    except Exception as e:
        print(f"Unexpected error in query: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
