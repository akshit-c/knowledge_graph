#!/usr/bin/env python3
"""
Research-grade plotting + analysis for MindLayer eval results.

Reads:
  - backend/eval/results_100.jsonl   (JSONL; one record per query)
  - backend/eval/metrics.json        (optional; used for cross-check / reporting)

Outputs (PDF + PNG @300dpi):
  - fig_confusion_matrix
  - fig_roc_curve
  - fig_pr_curve
  - fig_latency_violin
  - fig_score_scatter_regression
  - table_summary.csv
  - table_summary.md
  - per_query.csv

Usage:
  python backend/eval/plot_results.py \
    --results backend/eval/results_100.jsonl \
    --metrics backend/eval/metrics.json \
    --outdir backend/eval/figures

Notes:
- This script is robust to slightly different JSONL schemas.
- For ROC/PR it tries to extract a continuous "decision score" (kg/faiss/etc).
  If not found, it falls back to 0/1 predictions (AUC will be less informative).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn is standard for publishable ROC/PR; install if missing:
# pip install scikit-learn
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

NOT_IN_MEMORY_STRINGS = {"NOT_IN_MEMORY", "NOT_IN_CONTEXT", "NOT FOUND", "UNKNOWN"}


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str) and x.strip() != "":
            return float(x.strip())
        return None
    except Exception:
        return None


def get_nested(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def normalize_answer(ans: Any) -> str:
    if ans is None:
        return ""
    s = str(ans).strip()
    # strip common special tokens
    for tok in ["<|end|>", "</s>", "<|eot_id|>"]:
        s = s.replace(tok, "").strip()
    return s


def is_refusal_record(row: Dict[str, Any]) -> bool:
    """
    Determine if model refused.
    We treat NOT_IN_MEMORY as refusal.
    """
    # Explicit fields
    for k in ["refused", "is_refusal", "refusal"]:
        if k in row and isinstance(row[k], bool):
            return bool(row[k])

    ans = normalize_answer(row.get("answer") or row.get("prediction") or row.get("output"))
    if ans in NOT_IN_MEMORY_STRINGS:
        return True
    return False


def is_answerable_gold(row: Dict[str, Any]) -> Optional[bool]:
    """
    Determine gold label: answerable (positive) vs unanswerable (negative).
    Tries multiple common schemas.

    Returns:
      True/False if found, else None.
    """
    # Common labels
    for k in ["is_answerable", "answerable", "has_answer", "gold_answerable", "positive"]:
        if k in row and isinstance(row[k], bool):
            return bool(row[k])

    # If dataset contains gold answer (string) where NOT_IN_MEMORY indicates unanswerable
    for k in ["gold", "gold_answer", "expected", "expected_answer", "target"]:
        if k in row and row[k] is not None:
            s = normalize_answer(row[k])
            if s != "":
                return s not in NOT_IN_MEMORY_STRINGS

    # If labels are 0/1
    for k in ["label", "y_true", "gold_label"]:
        if k in row:
            v = row[k]
            if isinstance(v, (int, float)):
                return bool(int(v) == 1)
            if isinstance(v, str) and v.strip().isdigit():
                return bool(int(v.strip()) == 1)

    return None


def extract_latency_s(row: Dict[str, Any]) -> Optional[float]:
    # common places
    candidates = [
        row.get("latency_s"),
        row.get("latency"),
        row.get("time_s"),
        row.get("duration_s"),
        get_nested(row, "trace", "latency_s"),
        get_nested(row, "trace", "latency"),
    ]
    for c in candidates:
        f = safe_float(c)
        if f is not None and f >= 0:
            return f
    return None


def extract_decision_score(row: Dict[str, Any]) -> Optional[float]:
    """
    Extract a continuous score used for ranking/confidence.
    This is crucial for ROC/PR curves.
    Priority:
      1) explicit decision scores
      2) KG scores
      3) FAISS max score
      4) any retrieval score-like field
    """
    # Top-level direct candidates
    keys = [
        "decision_score",
        "score",
        "kg_score",
        "kg_max_score",
        "faiss_max_score",
        "max_score",
        "retrieval_score",
        "similarity",
        "confidence",
    ]
    for k in keys:
        f = safe_float(row.get(k))
        if f is not None:
            return f

    # trace dict candidates
    trace_keys = [
        ("trace", "decision_score"),
        ("trace", "kg_score"),
        ("trace", "kg_max_score"),
        ("trace", "faiss_max_score"),
        ("trace", "faiss_hits_max_score"),
        ("trace", "max_score"),
        ("trace", "retrieval_score"),
    ]
    for path in trace_keys:
        f = safe_float(get_nested(row, *path))
        if f is not None:
            return f

    # If sources contains kg_score
    sources = row.get("sources")
    if isinstance(sources, list) and sources:
        # Try max kg_score in sources
        svals = []
        for s in sources:
            if isinstance(s, dict):
                for kk in ["kg_score", "score", "similarity"]:
                    fv = safe_float(s.get(kk))
                    if fv is not None:
                        svals.append(fv)
        if svals:
            return float(max(svals))

    return None


@dataclass
class DerivedRow:
    idx: int
    question: str
    y_true_answerable: Optional[bool]
    y_pred_answerable: bool
    refused: bool
    answer: str
    latency_s: Optional[float]
    decision_score: Optional[float]


def load_results_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def derive_rows(raw: List[Dict[str, Any]]) -> List[DerivedRow]:
    derived: List[DerivedRow] = []
    for i, r in enumerate(raw):
        q = str(r.get("question") or r.get("query") or r.get("message") or "").strip()
        if not q:
            q = f"(missing_question_{i})"
        refused = is_refusal_record(r)
        y_pred_answerable = not refused

        y_true = is_answerable_gold(r)  # may be None
        ans = normalize_answer(r.get("answer") or r.get("prediction") or r.get("output"))

        latency_s = extract_latency_s(r)
        score = extract_decision_score(r)

        derived.append(
            DerivedRow(
                idx=i,
                question=q,
                y_true_answerable=y_true,
                y_pred_answerable=y_pred_answerable,
                refused=refused,
                answer=ans,
                latency_s=latency_s,
                decision_score=score,
            )
        )
    return derived


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, outdir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix (0=Unanswerable, 1=Answerable)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    # Annotate counts
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(outdir / f"fig_confusion_matrix.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_roc_pr(
    y_true: np.ndarray,
    scores: np.ndarray,
    outdir: Path,
) -> Tuple[Optional[float], Optional[float]]:
    # ROC
    roc_auc = None
    pr_auc = None

    # ROC requires both classes present
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        fig.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(outdir / f"fig_roc_curve.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # PR
        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(recall, precision, label=f"AP = {pr_auc:.3f}")
        ax.set_title("Precision–Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower left")
        fig.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(outdir / f"fig_pr_curve.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return roc_auc, pr_auc


def save_latency_violin(latencies: np.ndarray, outdir: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.violinplot(latencies, showmeans=True, showmedians=True, showextrema=True)
    ax.set_title("Latency Distribution (seconds)")
    ax.set_ylabel("Seconds")
    ax.set_xticks([1])
    ax.set_xticklabels(["/chat latency"])
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(outdir / f"fig_latency_violin.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_scatter_regression(
    scores: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    outdir: Path,
) -> None:
    """
    Scatter of decision score vs correctness (1 if predicted answerability matches gold).
    Adds regression line + Pearson correlation.
    """
    correctness = (y_true == y_pred).astype(int)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(scores, correctness)

    # Regression line (least squares)
    if len(scores) >= 2 and np.std(scores) > 1e-9:
        m, b = np.polyfit(scores, correctness, 1)
        xs = np.linspace(float(np.min(scores)), float(np.max(scores)), 200)
        ys = m * xs + b
        ax.plot(xs, ys)

        # Pearson r
        r = float(np.corrcoef(scores, correctness)[0, 1])
        ax.set_title(f"Decision Score vs Correctness (Pearson r = {r:.3f})")
    else:
        ax.set_title("Decision Score vs Correctness")

    ax.set_xlabel("Decision Score (KG/FAISS/Confidence)")
    ax.set_ylabel("Correctness (1=correct, 0=incorrect)")
    ax.set_yticks([0, 1])
    ax.set_ylim([-0.05, 1.05])

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(outdir / f"fig_score_scatter_regression.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    # Simple markdown table without external deps
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, required=True, help="Path to results_*.jsonl")
    ap.add_argument("--metrics", type=str, default="", help="Optional path to metrics.json")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for figures/tables")
    args = ap.parse_args()

    results_path = Path(args.results)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    raw = load_results_jsonl(results_path)
    derived = derive_rows(raw)

    # Build per-query dataframe
    df = pd.DataFrame([{
        "idx": d.idx,
        "question": d.question,
        "gold_answerable": d.y_true_answerable,
        "pred_answerable": d.y_pred_answerable,
        "refused": d.refused,
        "answer": d.answer,
        "latency_s": d.latency_s,
        "decision_score": d.decision_score,
    } for d in derived])

    # Save per-query CSV for appendix / reproducibility
    df.to_csv(outdir / "per_query.csv", index=False)

    # Filter rows where we have gold labels
    df_labeled = df[df["gold_answerable"].notna()].copy()
    if df_labeled.empty:
        raise SystemExit(
            "No gold labels found in results JSONL. "
            "Add a field like `is_answerable` or `label` or `gold` per line."
        )

    # y_true: 1=answerable, 0=unanswerable
    y_true = df_labeled["gold_answerable"].astype(int).to_numpy()
    y_pred = df_labeled["pred_answerable"].astype(int).to_numpy()

    # Determine scores for ROC/PR
    scores = df_labeled["decision_score"].to_numpy()
    have_scores = np.isfinite(scores).all() and not np.all(scores == scores[0])  # not constant
    if not have_scores:
        # Fall back: use predicted label as score (still produces curves, but less informative)
        scores = y_pred.astype(float)

    # Confusion matrix
    save_confusion_matrix(y_true, y_pred, outdir)

    # ROC & PR
    roc_auc, pr_auc = save_roc_pr(y_true, scores.astype(float), outdir)

    # Latency violin (all rows with latency)
    lat = df["latency_s"].dropna().to_numpy()
    if len(lat) > 0:
        save_latency_violin(lat.astype(float), outdir)

    # Score scatter/regression (requires labeled + finite scores)
    finite_mask = np.isfinite(scores)
    if finite_mask.sum() >= 3:
        save_scatter_regression(
            scores[finite_mask].astype(float),
            y_true[finite_mask].astype(int),
            y_pred[finite_mask].astype(int),
            outdir,
        )

    # Summary table (publishable)
    N = int(len(df_labeled))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Refusal Accuracy (RA): fraction of unanswerable that were refused => tn/(tn+fp)
    ra = (tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    # Hit Rate @k isn't directly available unless stored; we report classification instead.
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else float("nan")

    latency_mean = float(np.mean(lat)) if len(lat) else float("nan")
    latency_p95 = float(np.percentile(lat, 95)) if len(lat) else float("nan")

    summary = pd.DataFrame([{
        "N_labeled": N,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "RefusalAccuracy_RA": round(ra, 3) if np.isfinite(ra) else ra,
        "Precision": round(precision, 3) if np.isfinite(precision) else precision,
        "Recall": round(recall, 3) if np.isfinite(recall) else recall,
        "F1": round(f1, 3) if np.isfinite(f1) else f1,
        "ROC_AUC": round(roc_auc, 3) if roc_auc is not None else None,
        "PR_AUC_AP": round(pr_auc, 3) if pr_auc is not None else None,
        "Latency_mean_s": round(latency_mean, 3) if np.isfinite(latency_mean) else latency_mean,
        "Latency_p95_s": round(latency_p95, 3) if np.isfinite(latency_p95) else latency_p95,
        "Scores_used": "decision_score" if have_scores else "fallback_pred_label",
    }])

    summary.to_csv(outdir / "table_summary.csv", index=False)
    (outdir / "table_summary.md").write_text(markdown_table(summary), encoding="utf-8")

    # Optional: print and also compare to metrics.json if provided
    print("\nSaved figures + tables to:", outdir)
    print(summary.to_string(index=False))

    if args.metrics:
        mp = Path(args.metrics)
        if mp.exists():
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
                print("\nLoaded metrics.json (for reference):")
                print(json.dumps(m, indent=2))
            except Exception:
                pass


if __name__ == "__main__":
    main()
