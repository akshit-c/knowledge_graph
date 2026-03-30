import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import joblib


GATE_DIR = Path("backend/eval/gate")
MODEL_PATH = GATE_DIR / "refusal_gate_logreg.joblib"
THRESH_PATH = GATE_DIR / "refusal_gate_threshold.json"


def _tokenize(text: str) -> List[str]:
    import re

    STOPWORDS = set(
        "a an the and or but if then else is are was were be been being to of in on for with without from by as at into over under".split()
    )
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    toks = [t for t in toks if len(t) > 2 and t not in STOPWORDS]
    return toks


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def load_gate():
    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    feature_cols = payload["feature_cols"]

    thr_data = json.loads(THRESH_PATH.read_text())
    thr = float(thr_data.get("threshold"))
    return model, feature_cols, thr


def build_gate_features(
    question: str,
    trace: Dict[str, Any],
    sources: List[Dict[str, Any]],
) -> Dict[str, float]:
    # Extract scores
    faiss_max = float(trace.get("faiss_max_score") or 0.0)
    faiss_hits = float(trace.get("faiss_hits") or len(sources) or 0.0)

    # If sources contain FAISS scores, use them to refine
    src_scores = [float(s.get("score")) for s in sources if "score" in s]
    if faiss_max == 0.0 and src_scores:
        faiss_max = max(src_scores)
    faiss_gap = 0.0
    if len(src_scores) >= 2:
        top2 = sorted(src_scores)[-2:]
        faiss_gap = float(top2[-1] - top2[-2])

    kg_best = 0.0
    kg_support = 0.0
    for s in sources:
        if "kg_score" in s:
            kg_best = max(kg_best, float(s.get("kg_score") or 0.0))
        if "support" in s:
            kg_support = max(kg_support, float(s.get("support") or 0.0))

    kg_claims = float(trace.get("kg_claims") or 0.0)

    # Evidence text used
    evidence_text_chunks: List[str] = []
    for s in sources:
        ev = s.get("evidence")
        if isinstance(ev, list):
            for e in ev:
                t = (e or {}).get("text", "")
                if t:
                    evidence_text_chunks.append(t)
        t2 = s.get("text", "")
        if t2:
            evidence_text_chunks.append(t2)

    evidence_text = "\n".join(evidence_text_chunks)
    keyword_cov = _jaccard(_tokenize(question), _tokenize(evidence_text))
    evidence_tokens = float(len(_tokenize(evidence_text)))
    num_sources = float(len(sources) or 0.0)

    return {
        "faiss_max": faiss_max,
        "faiss_gap": faiss_gap,
        "faiss_hits": faiss_hits,
        "kg_best": kg_best,
        "kg_support": kg_support,
        "kg_claims": kg_claims,
        "keyword_cov": keyword_cov,
        "evidence_tokens": evidence_tokens,
        "num_sources": num_sources,
    }


def gate_decision(
    question: str,
    trace: Dict[str, Any],
    sources: List[Dict[str, Any]],
) -> Tuple[float, float, bool, Dict[str, float]]:
    """
    Returns:
      p: probability query is answerable
      thr: threshold used
      ok_to_answer: True if p >= thr
      feats: feature dict (for logging/analysis)
    """
    model, cols, thr = load_gate()
    feats = build_gate_features(question, trace, sources)
    x = np.array([[feats[c] for c in cols]], dtype=np.float32)
    p = float(model.predict_proba(x)[0, 1])
    return p, thr, (p >= thr), feats

