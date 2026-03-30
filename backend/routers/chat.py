from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import math
import re
import traceback

from backend.services import memory_search
from backend.services.kg_retriever import fetch_kg_context
from backend.services.local_llm_mlx import generate
from backend.services.refusal_gate import gate_decision

NOT_IN_MEMORY = "NOT_IN_MEMORY"

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    top_k: int = 5
    kg_k: int = 10
    mode: str = "full"  # kept for compatibility


_WORD_RE = re.compile(r"\b\w+\b")


def _keyword_coverage(question: str, text: str) -> float:
    """
    Fraction of unique, non-trivial question tokens that appear in the text.
    Deterministic, simple, and research-safe.
    """
    q_tokens = {
        t.lower()
        for t in _WORD_RE.findall(question or "")
        if len(t) >= 3
    }
    if not q_tokens:
        return 0.0

    t_lower = (text or "").lower()
    hits = 0
    for tok in q_tokens:
        if tok in t_lower:
            hits += 1
    return hits / max(1, len(q_tokens))


def _build_evidence_context(evidence: List[Dict[str, Any]]) -> str:
    """
    Turn reranked evidence into a labeled, auditable block for the LLM.
    """
    blocks: List[str] = []
    for i, ev in enumerate(evidence, 1):
        doc_id = ev.get("doc_id", "")
        chunk_id = ev.get("chunk_id", "")
        txt = (ev.get("text") or "").strip()
        blocks.append(
            f"EVIDENCE {i} (doc_id={doc_id}, chunk_id={chunk_id}):\n"
            f"{txt}"
        )
    return "\n\n".join(blocks)


@router.post("/")
def chat(req: ChatRequest):
    try:
        q = (req.message or "").strip()
        if not q:
            return {"answer": "", "sources": [], "trace": {"reason": "empty_message"}}

        # -----------------------------
        # 1) Retrieve from FAISS + KG
        # -----------------------------
        top_k = max(1, req.top_k)
        kg_k = max(1, req.kg_k)

        faiss_results = memory_search.search(q, top_k=top_k)
        kg_results = fetch_kg_context(question=q, k=kg_k, evidence_k=2)

        # If nothing comes back at all, hard NOT_IN_MEMORY.
        if not faiss_results and not kg_results:
            return {
                "answer": NOT_IN_MEMORY,
                "sources": [],
                "trace": {
                    "reason": "no_retrieval_hits",
                    "faiss_results": 0,
                    "kg_claims": 0,
                    "mode_used": "hybrid_kg_faiss",
                },
            }

        # -----------------------------
        # 2) Build candidate evidence pool
        # -----------------------------
        evidence_items: List[Dict[str, Any]] = []

        # FAISS chunks
        for r in faiss_results:
            txt = r.get("text") or ""
            kw_cov = _keyword_coverage(q, txt)
            evidence_items.append(
                {
                    "source": "faiss",
                    "doc_id": r.get("doc_id"),
                    "chunk_id": r.get("chunk_id"),
                    "text": txt,
                    "faiss_score": float(r.get("score", 0.0)),
                    "keyword_cov": kw_cov,
                    "support": 0,
                    "kg_score": None,
                    "claim": None,
                }
            )

        # KG claim evidence chunks
        max_support = 0
        for r in kg_results or []:
            support = int(r.get("support") or 0)
            kg_score = float(r.get("kg_score") or 0.0)
            if support > max_support:
                max_support = support
            for ev in r.get("evidence") or []:
                txt = (ev.get("text") or "").strip()
                if not txt:
                    continue
                kw_cov = _keyword_coverage(q, txt)
                evidence_items.append(
                    {
                        "source": "kg",
                        "doc_id": ev.get("doc_id"),
                        "chunk_id": ev.get("chunk_id"),
                        "text": txt,
                        "faiss_score": 0.0,
                        "keyword_cov": kw_cov,
                        "support": support,
                        "kg_score": kg_score,
                        "claim": r.get("claim"),
                    }
                )

        # -----------------------------
        # 3) Rerank: score = 0.65 * faiss_score + 0.35 * keyword_coverage
        #    If KG: + 0.15 * log(1 + support)
        # -----------------------------
        for ev in evidence_items:
            base = 0.65 * float(ev.get("faiss_score", 0.0)) + 0.35 * float(
                ev.get("keyword_cov", 0.0)
            )
            if ev.get("source") == "kg":
                support = max(0, int(ev.get("support") or 0))
                base += 0.15 * math.log(1.0 + support)
            ev["score"] = base

        evidence_items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        top_evidence = evidence_items[:10]

        # -----------------------------
        # 4) Logistic refusal gate (aggressive FP<=1)
        # -----------------------------
        if not top_evidence:
            return {
                "answer": NOT_IN_MEMORY,
                "sources": [],
                "trace": {
                    "reason": "no_evidence_after_rerank",
                    "faiss_results": len(faiss_results),
                    "kg_claims": len(kg_results or []),
                    "max_support": max_support,
                    "mode_used": "hybrid_kg_faiss_logreg_gate",
                },
            }

        # Build base trace used by the gate and for logging.
        faiss_max_score = 0.0
        faiss_max_kw = 0.0
        if faiss_results:
            best_faiss = max(faiss_results, key=lambda r: float(r.get("score", 0.0)))
            faiss_max_score = float(best_faiss.get("score", 0.0))
            faiss_max_kw = _keyword_coverage(q, best_faiss.get("text") or "")

        trace: Dict[str, Any] = {
            "kg_claims": len(kg_results or []),
            "faiss_results": len(faiss_results),
            "max_support": max_support,
            "faiss_max_score": faiss_max_score,
            "faiss_max_kw": faiss_max_kw,
        }

        # For metrics (ROC/PR), expose a scalar score as kg_score on the top evidence.
        if top_evidence:
            top_evidence[0]["kg_score"] = float(top_evidence[0].get("score", 0.0))

        # Logistic gate over features derived from retrieval-only signals.
        p, thr, ok_to_answer, feats = gate_decision(q, trace, top_evidence)
        trace["gate_p_answerable"] = p
        trace["gate_threshold"] = thr
        trace["gate_features"] = feats
        trace["gate_policy"] = "fp_budget=1_aggressive"

        if not ok_to_answer:
            # Refuse without calling MLX; still return top evidence for audits.
            sources_payload: List[Dict[str, Any]] = []
            for ev in top_evidence:
                sources_payload.append(
                    {
                        "doc_id": ev.get("doc_id"),
                        "chunk_id": ev.get("chunk_id"),
                        "text": ev.get("text"),
                        "source": ev.get("source"),
                        "faiss_score": ev.get("faiss_score"),
                        "keyword_coverage": ev.get("keyword_cov"),
                        "support": ev.get("support"),
                        "score": ev.get("score"),
                        "claim": ev.get("claim"),
                        "kg_score": ev.get("kg_score"),
                    }
                )

            trace["gate_reason"] = "logreg_refuse"
            trace["mode_used"] = "hybrid_kg_faiss_logreg_gate"

            return {
                "answer": NOT_IN_MEMORY,
                "sources": sources_payload,
                "trace": trace,
            }

        # -----------------------------
        # 5) Extract-then-answer prompting (only if gate passes)
        # -----------------------------
        context = _build_evidence_context(top_evidence)

        prompt = f"""
You are a grounded AI assistant.

You are given a user question and a set of evidence snippets from the user's own documents.

First, COPY the exact sentence or sentences from the evidence that answer the question.
Then, on a new line starting with \"ANSWER:\", write a single-sentence answer in your own words.

If NONE of the evidence actually answers the question, output exactly: {NOT_IN_MEMORY}.

EVIDENCE:
{context}

QUESTION:
{q}
""".strip()

        answer = generate(prompt).strip()

        # Prepare lightweight sources payload (include scores for eval/debug)
        sources_payload: List[Dict[str, Any]] = []
        for ev in top_evidence:
            payload = {
                "doc_id": ev.get("doc_id"),
                "chunk_id": ev.get("chunk_id"),
                "text": ev.get("text"),
                "source": ev.get("source"),
                "faiss_score": ev.get("faiss_score"),
                "keyword_coverage": ev.get("keyword_cov"),
                "support": ev.get("support"),
                "score": ev.get("score"),
                "claim": ev.get("claim"),
                "kg_score": ev.get("kg_score"),
            }
            sources_payload.append(payload)

        trace["gate_reason"] = "logreg_allow"
        trace["mode_used"] = "hybrid_kg_faiss_logreg_gate"

        return {
            "answer": answer,
            "sources": sources_payload,
            "trace": trace,
        }

    except Exception as e:
        print("CHAT ERROR:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
