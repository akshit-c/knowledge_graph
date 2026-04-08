#!/usr/bin/env python3
"""
Run MindLayer eval under multiple system modes and write comparative results.

This is meant to generate paper-ready tables from:
  - backend/eval/run_eval.py (per-query + summary.json)

Modes:
  - baseline: FAISS-only retrieval, no refusal gate
  - hybrid: FAISS+KG retrieval, no refusal gate
  - calibrated: FAISS+KG retrieval + calibrated refusal gate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fmt_float(x: Any, ndigits: int = 3) -> str:
    try:
        if x is None:
            return "-"
        if isinstance(x, (float, int)):
            return f"{float(x):.{ndigits}f}"
    except Exception:
        pass
    return str(x)


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows: List[str] = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(_fmt_float(r[c]) for c in cols) + " |")

    return "\n".join([header, sep] + rows)


def run_eval_for_mode(
    *,
    endpoint: str,
    queries_path: str,
    outbase: str,
    top_k: int,
    kg_k: int,
    mode: str,
    sleep_s: float,
    skip_if_summary_exists: bool,
) -> Path:
    """
    Returns the path to summary.json for the run.
    """
    outdir = Path(outbase) / mode
    summary_path = outdir / "summary.json"
    if skip_if_summary_exists and summary_path.exists():
        return summary_path

    cmd = [
        sys.executable,
        str(_repo_root() / "backend/eval/run_eval.py"),
        "--endpoint",
        endpoint,
        "--queries",
        queries_path,
        "--outdir",
        outbase,
        "--top_k",
        str(top_k),
        "--kg_k",
        str(kg_k),
        "--mode",
        mode,
        "--sleep",
        str(sleep_s),
    ]
    subprocess.run(cmd, check=True)
    return summary_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/chat/", help="FastAPI /chat/ endpoint")
    ap.add_argument("--queries", default="backend/eval/queries.jsonl", help="Path to queries.jsonl")
    ap.add_argument("--outbase", default="backend/eval/comparative_runs", help="Output base directory")
    ap.add_argument("--figures-outdir", default="backend/eval/figures", help="Where to write comparative tables")
    ap.add_argument("--top_k", type=int, default=5, help="Retriever top_k for run_eval.py")
    ap.add_argument("--kg_k", type=int, default=10, help="KG top claims for run_eval.py")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep between requests")
    ap.add_argument("--skip-if-summary-exists", action="store_true", help="Reuse existing run outputs")
    args = ap.parse_args()

    modes = ["baseline", "hybrid", "calibrated"]

    rows: List[Dict[str, Any]] = []
    for mode in modes:
        summary_path = run_eval_for_mode(
            endpoint=args.endpoint,
            queries_path=args.queries,
            outbase=args.outbase,
            top_k=args.top_k,
            kg_k=args.kg_k,
            mode=mode,
            sleep_s=args.sleep,
            skip_if_summary_exists=args.skip_if_summary_exists,
        )

        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        rows.append(
            {
                "Mode": mode,
                "N": summary.get("N"),
                "HR@k": summary.get("HR@k_mean"),
                "RA": summary.get("RA_mean"),
                "GAA": summary.get("GAA_mean"),
                "EM (exact)": summary.get("EM_mean"),
                "F1 (token overlap)": summary.get("F1_token_overlap_mean"),
                "Evidence Precision": summary.get("Evidence_precision_mean"),
                "Evidence Recall": summary.get("Evidence_recall_mean"),
                "Evidence F1": summary.get("Evidence_f1_mean"),
                "Faithfulness": summary.get("Faithfulness_mean"),
                "Attribution Accuracy": summary.get("Attribution_accuracy_mean"),
                "Hallucination Rate": summary.get("Hallucination_rate_mean"),
                "Latency (mean, s)": summary.get("Latency_mean_s"),
            }
        )

    df = pd.DataFrame(rows)

    outdir = Path(args.figures_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "comparative_results.csv"
    md_path = outdir / "comparative_results.md"

    df.to_csv(csv_path, index=False)
    md_path.write_text(_markdown_table(df), encoding="utf-8")

    print("✅ Comparative results written to:")
    print(f" - {csv_path}")
    print(f" - {md_path}")


if __name__ == "__main__":
    main()

