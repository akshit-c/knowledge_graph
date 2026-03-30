import json
import re
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


RESULTS_PATH = Path("backend/eval/results_100.jsonl")
OUT_DIR = Path("backend/eval/gate")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "refusal_gate_logreg.joblib"
THRESH_PATH = OUT_DIR / "refusal_gate_threshold.json"
FEATURES_CSV = OUT_DIR / "features.csv"


STOPWORDS = set("""
a an the and or but if then else is are was were be been being
to of in on for with without from by as at into over under
this that these those it its i you we they he she them us
""".split())


def safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def tokenize(text: str) -> List[str]:
    text = text.lower()
    toks = re.findall(r"[a-z0-9]+", text)
    toks = [t for t in toks if len(t) > 2 and t not in STOPWORDS]
    return toks


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def pick_evidence_text(rec: Dict[str, Any]) -> str:
    """
    Tries to build the evidence text the model used:
    - KG mode: sources[*].evidence[*].text
    - FAISS mode: sources[*].text
    """
    parts = []
    for s in rec.get("sources", []) or []:
        # KG sources often look like: {"claim":..., "evidence":[{"text":...}, ...]}
        ev = s.get("evidence")
        if isinstance(ev, list):
            for e in ev:
                t = (e or {}).get("text", "")
                if t:
                    parts.append(t)
        # FAISS sources often look like: {"text": "...", "score": ...}
        t2 = s.get("text", "")
        if t2:
            parts.append(t2)
    return "\n".join(parts).strip()


def extract_scores(rec: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts best-effort numeric signals from:
    - trace (faiss_max_score, kg_claims, etc.)
    - sources scores (faiss score, kg_score, support)
    """
    trace = rec.get("trace") or {}

    # FAISS similarity signals
    faiss_max = safe_float(trace.get("faiss_max_score"), default=np.nan)
    faiss_hits = safe_float(trace.get("faiss_hits"), default=np.nan)

    # If trace missing, infer from sources[*].score
    src_scores = []
    for s in rec.get("sources", []) or []:
        if "score" in s:
            src_scores.append(safe_float(s.get("score"), default=np.nan))
    if (math.isnan(faiss_max) or faiss_max == 0.0) and src_scores:
        faiss_max = np.nanmax(src_scores)
    faiss_top2 = np.sort([x for x in src_scores if not math.isnan(x)])[-2:] if len(src_scores) >= 2 else []
    faiss_gap = float(faiss_top2[-1] - faiss_top2[-2]) if len(faiss_top2) == 2 else 0.0

    # KG signals
    kg_claims = safe_float(trace.get("kg_claims"), default=np.nan)
    kg_best = np.nan
    kg_support = 0.0

    for s in rec.get("sources", []) or []:
        if "kg_score" in s:
            kg_best = max(kg_best if not math.isnan(kg_best) else -1e9, safe_float(s.get("kg_score"), default=np.nan))
        if "support" in s:
            kg_support = max(kg_support, safe_float(s.get("support"), default=0.0))

    if math.isnan(kg_best):
        kg_best = 0.0
    if math.isnan(kg_claims):
        kg_claims = 0.0
    if math.isnan(faiss_hits):
        faiss_hits = float(len(rec.get("sources", []) or []))
    if math.isnan(faiss_max):
        faiss_max = 0.0

    return {
        "faiss_max": float(faiss_max),
        "faiss_gap": float(faiss_gap),
        "faiss_hits": float(faiss_hits),
        "kg_best": float(kg_best),
        "kg_support": float(kg_support),
        "kg_claims": float(kg_claims),
    }


def infer_label(rec: Dict[str, Any]) -> int:
    """
    Label = 1 if query is answerable (ground-truth positive), else 0.
    We infer from eval set fields. Supports multiple conventions.
    """
    # Common fields in eval jsonl
    # expected_label: 1/0, or "positive"/"negative"
    if "expected_label" in rec:
        v = rec["expected_label"]
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            return 1 if v.lower() in ("pos", "positive", "answerable", "true", "1") else 0

    # Our eval_100 format: label is "POS"/"NEG"
    if "label" in rec and isinstance(rec["label"], str):
        return 1 if rec["label"].upper() == "POS" else 0

    # If your generator stores "type": "pos"/"neg"
    if "type" in rec and isinstance(rec["type"], str):
        return 1 if rec["type"].lower().startswith("pos") else 0

    # If it stores "is_answerable": true/false
    if "is_answerable" in rec:
        return 1 if bool(rec["is_answerable"]) else 0

    # Fallback: if expected contains NOT_IN_MEMORY, treat as negative,
    # and treat explicit "ANSWER" as positive.
    exp = (rec.get("expected") or rec.get("expected_answer") or "")
    if isinstance(exp, str):
        exp_clean = exp.strip().upper()
        if exp_clean == "NOT_IN_MEMORY":
            return 0
        if exp_clean == "ANSWER":
            return 1

    raise ValueError("Could not infer label from record. Add expected_label/type/is_answerable to results JSONL.")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_feature_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    feats = []
    for i, rec in enumerate(records):
        q = rec.get("question") or rec.get("message") or rec.get("query") or ""
        ans = (rec.get("answer") or "").strip()
        evidence_text = pick_evidence_text(rec)

        q_toks = tokenize(q)
        e_toks = tokenize(evidence_text)

        scores = extract_scores(rec)

        # Coverage and evidence quality
        keyword_cov = jaccard(q_toks, e_toks)
        evidence_len = float(len(evidence_text))
        evidence_tokens = float(len(e_toks))
        num_sources = float(len(rec.get("sources", []) or []))

        # Was model refusing?
        refused = 1 if ans == "NOT_IN_MEMORY" else 0

        # Latency if available
        latency = safe_float(rec.get("latency_s") or (rec.get("trace") or {}).get("latency_s"), default=np.nan)
        if math.isnan(latency):
            latency = 0.0

        y = infer_label(rec)

        feats.append({
            "id": rec.get("id", i),
            "question": q,
            "answer": ans,
            "label_answerable": y,
            "refused": refused,
            "keyword_cov": keyword_cov,
            "evidence_len": evidence_len,
            "evidence_tokens": evidence_tokens,
            "num_sources": num_sources,
            "latency_s": latency,
            **scores,
        })

    df = pd.DataFrame(feats)
    return df


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_png: Path) -> float:
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={auc:.3f})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return auc


def plot_pr(y_true: np.ndarray, y_prob: np.ndarray, out_png: Path) -> float:
    ap = average_precision_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP={ap:.3f})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return ap


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, out_png: Path) -> float:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_true, y_prob)
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration (Brier={brier:.3f})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return brier


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray, fp_budget: int = 0) -> float:
    """
    Choose the LOWEST threshold that keeps FP <= fp_budget,
    which maximizes recall under hallucination constraint.
    """
    thresholds = np.unique(y_prob)
    thresholds.sort()

    best = None
    best_recall = -1.0

    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        if fp <= fp_budget:
            recall = tp / max(1, (tp + fn))
            if recall > best_recall:
                best_recall = recall
                best = float(t)

    if best is None:
        # If impossible, fallback to conservative threshold
        best = 0.99
    return best


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing {RESULTS_PATH}. Run evaluation first to create results_100.jsonl")

    records = load_jsonl(RESULTS_PATH)
    df = build_feature_frame(records)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"Saved features: {FEATURES_CSV} ({len(df)} rows)")

    # Features for the gate: only signals available BEFORE generation.
    feature_cols = [
        "faiss_max", "faiss_gap", "faiss_hits",
        "kg_best", "kg_support", "kg_claims",
        "keyword_cov",
        "evidence_tokens",
        "num_sources",
    ]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label_answerable"].values.astype(int)

    # Train final model
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X, y)
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")

    # In-sample probabilities (you can add proper CV reporting below)
    y_prob = model.predict_proba(X)[:, 1]

    auc = plot_roc(y, y_prob, OUT_DIR / "roc.png")
    ap = plot_pr(y, y_prob, OUT_DIR / "pr.png")
    brier = plot_calibration(y, y_prob, OUT_DIR / "calibration.png")

    # Pick threshold for strict RA: fp_budget=0 means "no hallucinations allowed"
    thr0 = choose_threshold(y, y_prob, fp_budget=0)
    thr1 = choose_threshold(y, y_prob, fp_budget=1)

    def report(thr: float, name: str):
        y_hat = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = (2 * precision * recall) / max(1e-9, (precision + recall))

        # For refusal accuracy RA (negatives correctly refused):
        # If negative label=0, correct refusal means predict 0.
        ra = tn / max(1, (tn + fp))

        print(f"\n=== THRESHOLD {name} ===")
        print(f"thr = {thr:.6f}")
        print(f"Confusion [[TN, FP], [FN, TP]] = [[{tn}, {fp}], [{fn}, {tp}]]")
        print(f"RA (Refusal Accuracy) = {ra:.3f}")
        print(f"Precision = {precision:.3f}")
        print(f"Recall = {recall:.3f}")
        print(f"F1 = {f1:.3f}")

    print(f"\nROC-AUC = {auc:.3f} | PR-AP = {ap:.3f} | Brier = {brier:.3f}")
    report(thr0, "FP<=0 (paper-safe)")
    report(thr1, "FP<=1 (aggressive)")

    # Save both thresholds; choose aggressive FP<=1 as default for experimentation
    chosen = {
        "threshold": thr1,
        "policy": "fp_budget=1",
        "thresholds": {
            "fp_le_0": thr0,
            "fp_le_1": thr1,
        },
        "feature_cols": feature_cols,
        "roc_auc": auc,
        "pr_ap": ap,
        "brier": brier,
    }
    THRESH_PATH.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(f"\nSaved threshold policy: {THRESH_PATH}")

    # Also save coefficients (paper-friendly)
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    coef_df = pd.DataFrame({"feature": feature_cols, "coef": coef})
    coef_df.to_csv(OUT_DIR / "coefficients.csv", index=False)
    (OUT_DIR / "intercept.txt").write_text(str(intercept), encoding="utf-8")
    print(f"Saved coefficients: {OUT_DIR / 'coefficients.csv'}")


if __name__ == "__main__":
    main()
