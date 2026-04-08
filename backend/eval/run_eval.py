import argparse
import json
import time
import re
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


NOT_IN_MEMORY = "NOT_IN_MEMORY"


@dataclass
class EvalItem:
    id: str
    query: str
    label: str  # "in_memory" | "not_in_memory"
    gold_answer: str
    gold_doc_ids: List[str]
    gold_chunk_ids: List[str]
    gold_evidence_phrases: List[str]


def load_queries(path: Path) -> List[EvalItem]:
    items: List[EvalItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                EvalItem(
                    id=obj["id"],
                    query=obj["query"],
                    label=obj["label"],
                    gold_answer=obj.get("gold_answer", ""),
                    gold_doc_ids=obj.get("gold_doc_ids", []) or [],
                    gold_chunk_ids=obj.get("gold_chunk_ids", []) or [],
                    gold_evidence_phrases=obj.get("gold_evidence_phrases", []) or [],
                )
            )
    return items


def normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def answer_is_refusal(ans: str) -> bool:
    return normalize_text(ans) == normalize_text(NOT_IN_MEMORY)


def answer_matches_gold(ans: str, gold: str) -> bool:
    """
    Research-friendly but simple:
    - exact match OR gold is a substring of answer OR answer is a substring of gold.
    For journal submission, we can later add LLM-as-judge + human eval. But this is deterministic.
    """
    a = normalize_text(ans)
    g = normalize_text(gold)
    if not g:
        return False
    if a == g:
        return True
    if g in a:
        return True
    if a in g:
        return True
    return False


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize_for_overlap(s: str) -> List[str]:
    s = normalize_text(s or "")
    return _TOKEN_RE.findall(s)


def answer_exact_match(ans: str, gold: str) -> int:
    """
    Exact match under `normalize_text` (journal-friendly EM).
    """
    return int(normalize_text(ans) == normalize_text(gold))


def token_overlap_f1(ans: str, gold: str) -> float:
    """
    SQuAD-style token F1 (alphanumeric token overlap with multiset counts).
    """
    pred_toks = _tokenize_for_overlap(ans)
    gold_toks = _tokenize_for_overlap(gold)

    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0

    pred_counts = Counter(pred_toks)
    gold_counts = Counter(gold_toks)
    common = pred_counts & gold_counts
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / max(1, len(pred_toks))
    recall = num_same / max(1, len(gold_toks))
    return float(2 * precision * recall / max(1e-9, (precision + recall)))


def evidence_ids_and_text(
    item: EvalItem,
    retrieved_doc_ids: List[str],
    retrieved_chunk_ids: List[str],
    retrieved_texts: List[str],
) -> Tuple[List[str], List[str], str]:
    """
    Choose the evidence ID granularity to score against gold.
    Prefer chunk IDs when gold provides them, else fall back to doc IDs.
    """
    if item.gold_chunk_ids:
        gold_ids = item.gold_chunk_ids
        retrieved_ids = retrieved_chunk_ids
    elif item.gold_doc_ids:
        gold_ids = item.gold_doc_ids
        retrieved_ids = retrieved_doc_ids
    else:
        gold_ids = []
        retrieved_ids = retrieved_chunk_ids or retrieved_doc_ids

    evidence_text = "\n".join(retrieved_texts or []).strip()
    return gold_ids, retrieved_ids, evidence_text


def evidence_precision_recall_f1(
    item: EvalItem,
    retrieved_doc_ids: List[str],
    retrieved_chunk_ids: List[str],
    retrieved_texts: List[str],
) -> Tuple[float, float, float]:
    gold_ids, retrieved_ids, _ = evidence_ids_and_text(
        item, retrieved_doc_ids, retrieved_chunk_ids, retrieved_texts
    )
    gold_set = set(gold_ids or [])
    retrieved_set = set(retrieved_ids or [])
    num_retrieved = len(retrieved_set)
    num_gold = len(gold_set)
    if num_retrieved == 0:
        return (0.0, 0.0, 0.0)

    num_correct = len(gold_set & retrieved_set)
    precision = num_correct / max(1, num_retrieved)
    recall = num_correct / max(1, num_gold) if num_gold > 0 else 0.0
    if precision == 0.0 and recall == 0.0:
        return (precision, recall, 0.0)
    f1 = 2 * precision * recall / max(1e-9, (precision + recall))
    return (float(precision), float(recall), float(f1))


def token_coverage(answer: str, evidence_text: str) -> float:
    """
    Faithfulness heuristic: fraction of answer tokens present in retrieved evidence text.
    """
    a_toks = set(_tokenize_for_overlap(answer))
    e_toks = set(_tokenize_for_overlap(evidence_text))
    if not a_toks:
        return 0.0
    return float(len(a_toks & e_toks) / max(1, len(a_toks)))


def hallucination_flag(item: EvalItem, answer: str, is_refusal: bool, hr: int) -> int:
    """
    Hallucination heuristic (lower is better):
    - if the query is answerable (in_memory), hallucinated when it answers but evidence hit is 0
    - if the query is unanswerable (not_in_memory), hallucinated when it answers (i.e., does not refuse)
    """
    if item.label == "in_memory":
        return int((not is_refusal) and hr == 0)
    return int((not is_refusal) and item.label == "not_in_memory")


def extract_source_info(sources: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract unique doc_ids, chunk_ids, and evidence texts from the /chat/ response.

    Supports multiple schemas:
    - `sources[*].evidence[*]` (older/alternate schema)
    - `sources[*].doc_id` / `sources[*].chunk_id` / `sources[*].text` (current /chat/ output)
    """
    doc_ids: List[str] = []
    chunk_ids: List[str] = []
    texts: List[str] = []

    for src in sources or []:
        evidences = src.get("evidence")
        # Schema A: nested evidence list.
        if isinstance(evidences, list) and evidences:
            for ev in evidences:
                if not isinstance(ev, dict):
                    continue
                did = ev.get("doc_id")
                cid = ev.get("chunk_id")
                txt = ev.get("text") or ""
                if did and did not in doc_ids:
                    doc_ids.append(did)
                if cid and cid not in chunk_ids:
                    chunk_ids.append(cid)
                if txt:
                    texts.append(txt)
            continue

        # Schema B: direct fields on each source entry.
        if not isinstance(src, dict):
            continue
        did = src.get("doc_id")
        cid = src.get("chunk_id")
        txt = src.get("text") or ""
        if did and did not in doc_ids:
            doc_ids.append(did)
        if cid and cid not in chunk_ids:
            chunk_ids.append(cid)
        if txt:
            texts.append(str(txt))

    return doc_ids, chunk_ids, texts


def compute_hit_rate(
    item: EvalItem,
    retrieved_doc_ids: List[str],
    retrieved_chunk_ids: List[str],
    retrieved_texts: List[str],
) -> int:
    """
    HR@k = 1 if ANY gold evidence appears in retrieved top-k.
    Prefer chunk_ids if present, else doc_ids.
    """
    if item.gold_chunk_ids:
        return int(any(g in retrieved_chunk_ids for g in item.gold_chunk_ids))
    if item.gold_doc_ids:
        return int(any(g in retrieved_doc_ids for g in item.gold_doc_ids))
    # ID-free HR: fall back to phrase hits in retrieved evidence text.
    if item.gold_evidence_phrases:
        norm_texts = [normalize_text(t) for t in retrieved_texts]
        for phrase in item.gold_evidence_phrases:
            p = normalize_text(phrase)
            if not p:
                continue
            if any(p in t for t in norm_texts):
                return 1
        return 0
    # If no gold evidence is provided, HR is undefined; treat as 0 to avoid inflating.
    return 0


def bootstrap_ci(values: List[int], iters: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """
    Bootstrap 95% CI for a binary metric (0/1).
    Returns (low, high) in [0,1].
    """
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    n = len(arr)
    means = []
    for _ in range(iters):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(sample.mean())
    means = np.array(means)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/chat/", help="FastAPI /chat/ endpoint")
    ap.add_argument("--queries", default="backend/eval/queries.jsonl", help="Path to queries.jsonl")
    ap.add_argument("--outdir", default="backend/eval/results", help="Output directory")
    ap.add_argument(
        "--mode",
        default="calibrated",
        choices=["baseline", "hybrid", "calibrated"],
        help="System mode to pass to /chat/ (controls retrieval + refusal gate).",
    )
    ap.add_argument("--top_k", type=int, default=5, help="Retriever top_k to request")
    ap.add_argument("--kg_k", type=int, default=10, help="KG top claims to request")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests (avoid overheating)")
    args = ap.parse_args()

    outdir = Path(args.outdir) / args.mode
    outdir.mkdir(parents=True, exist_ok=True)

    items = load_queries(Path(args.queries))
    if not items:
        raise SystemExit("No queries found. Fill backend/eval/queries.jsonl first.")

    log_path = outdir / "runs.jsonl"

    rows = []
    hr_values = []
    ra_values = []
    gaa_values = []

    em_values: List[int] = []
    f1_values: List[float] = []
    evidence_precision_values: List[float] = []
    evidence_recall_values: List[float] = []
    evidence_f1_values: List[float] = []
    faithfulness_values: List[float] = []
    attribution_accuracy_values: List[float] = []
    hallucination_values: List[int] = []

    with log_path.open("w", encoding="utf-8") as logf:
        for it in items:
            payload = {"message": it.query, "top_k": args.top_k, "kg_k": args.kg_k, "mode": args.mode}
            t0 = time.time()
            resp = requests.post(args.endpoint, json=payload, timeout=120)
            latency = time.time() - t0

            if resp.status_code != 200:
                data = {"error": f"HTTP {resp.status_code}", "body": resp.text}
                answer = ""
                sources = []
            else:
                data = resp.json()
                answer = (data.get("answer") or "").strip()
                sources = data.get("sources") or []

            retrieved_doc_ids, retrieved_chunk_ids, retrieved_texts = extract_source_info(sources)

            # HR@k: evidence hit (ID-based if available, else phrase-based)
            hr = compute_hit_rate(it, retrieved_doc_ids, retrieved_chunk_ids, retrieved_texts)
            hr_values.append(hr)

            # RA: refusal correctness
            # - if label=not_in_memory => should refuse
            # - if label=in_memory => should NOT refuse
            is_refusal = answer_is_refusal(answer)
            ra = int((it.label == "not_in_memory" and is_refusal) or (it.label == "in_memory" and not is_refusal))
            ra_values.append(ra)

            # GAA: grounded answer accuracy
            # For in_memory: correct answer AND evidence hit
            # For not_in_memory: correct refusal (and evidence hit is irrelevant)
            if it.label == "in_memory":
                correct = answer_matches_gold(answer, it.gold_answer)
                gaa = int(correct and hr == 1 and not is_refusal)
            else:
                gaa = int(is_refusal)  # must refuse
            gaa_values.append(gaa)

            # EM and token F1 over the gold answer string.
            em = answer_exact_match(answer, it.gold_answer)
            f1 = token_overlap_f1(answer, it.gold_answer)
            em_values.append(em)
            f1_values.append(f1)

            # Evidence quality / KG metrics.
            ev_prec, ev_rec, ev_f1 = evidence_precision_recall_f1(
                it, retrieved_doc_ids, retrieved_chunk_ids, retrieved_texts
            )
            evidence_precision_values.append(ev_prec)
            evidence_recall_values.append(ev_rec)
            evidence_f1_values.append(ev_f1)

            # Faithfulness + attribution (heuristic).
            # - Faithfulness is token coverage of the answer in retrieved evidence text.
            # - Attribution accuracy is precision of cited evidence (chunk/doc IDs) vs gold.
            _, _, evidence_text = evidence_ids_and_text(
                it, retrieved_doc_ids, retrieved_chunk_ids, retrieved_texts
            )
            if it.label == "not_in_memory":
                faithfulness = float(is_refusal)
                attribution_accuracy = 0.0
            else:
                faithfulness = float(token_coverage(answer, evidence_text)) if not is_refusal else 0.0
                attribution_accuracy = float(ev_prec) if (not is_refusal) else 0.0
            faithfulness_values.append(faithfulness)
            attribution_accuracy_values.append(attribution_accuracy)

            # Hallucination rate (heuristic).
            hallucinated = hallucination_flag(it, answer, is_refusal, hr)
            hallucination_values.append(hallucinated)

            row = {
                "id": it.id,
                "query": it.query,
                "label": it.label,
                "gold_answer": it.gold_answer,
                "answer": answer,
                "is_refusal": int(is_refusal),
                "hr@k": hr,
                "ra": ra,
                "gaa": gaa,
                "em": em,
                "f1_token_overlap": f1,
                "evidence_precision": ev_prec,
                "evidence_recall": ev_rec,
                "evidence_f1": ev_f1,
                "faithfulness": faithfulness,
                "attribution_accuracy": attribution_accuracy,
                "hallucination": hallucinated,
                "latency_s": latency,
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieved_chunk_ids": retrieved_chunk_ids,
            }
            rows.append(row)

            logf.write(json.dumps({"request": payload, "response": data, "row": row}, ensure_ascii=False) + "\n")

            if args.sleep > 0:
                time.sleep(args.sleep)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "per_query.csv", index=False)

    # Aggregate metrics
    metrics = {
        "N": len(df),
        "HR@k_mean": float(df["hr@k"].mean()),
        "RA_mean": float(df["ra"].mean()),
        "GAA_mean": float(df["gaa"].mean()),
        "EM_mean": float(df["em"].mean()),
        "F1_token_overlap_mean": float(df["f1_token_overlap"].mean()),
        "Evidence_precision_mean": float(df["evidence_precision"].mean()),
        "Evidence_recall_mean": float(df["evidence_recall"].mean()),
        "Evidence_f1_mean": float(df["evidence_f1"].mean()),
        "Faithfulness_mean": float(df["faithfulness"].mean()),
        "Attribution_accuracy_mean": float(df["attribution_accuracy"].mean()),
        "Hallucination_rate_mean": float(df["hallucination"].mean()),
        "Latency_mean_s": float(df["latency_s"].mean()),
        "Latency_p95_s": float(df["latency_s"].quantile(0.95)),
    }

    # Bootstrap CIs
    hr_ci = bootstrap_ci(hr_values)
    ra_ci = bootstrap_ci(ra_values)
    gaa_ci = bootstrap_ci(gaa_values)

    metrics_ci = {
        "HR@k_CI95_low": hr_ci[0],
        "HR@k_CI95_high": hr_ci[1],
        "RA_CI95_low": ra_ci[0],
        "RA_CI95_high": ra_ci[1],
        "GAA_CI95_low": gaa_ci[0],
        "GAA_CI95_high": gaa_ci[1],
    }

    summary = {**metrics, **metrics_ci}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Publication-ready table
    table_df = pd.DataFrame(
        [
            {
                "Metric": "HR@k",
                "Mean": summary["HR@k_mean"],
                "CI95": f"[{summary['HR@k_CI95_low']:.3f}, {summary['HR@k_CI95_high']:.3f}]",
            },
            {
                "Metric": "RA",
                "Mean": summary["RA_mean"],
                "CI95": f"[{summary['RA_CI95_low']:.3f}, {summary['RA_CI95_high']:.3f}]",
            },
            {
                "Metric": "GAA",
                "Mean": summary["GAA_mean"],
                "CI95": f"[{summary['GAA_CI95_low']:.3f}, {summary['GAA_CI95_high']:.3f}]",
            },
            {"Metric": "EM (exact)", "Mean": summary["EM_mean"], "CI95": "-"},
            {"Metric": "F1 (token overlap)", "Mean": summary["F1_token_overlap_mean"], "CI95": "-"},
            {
                "Metric": "Evidence Precision",
                "Mean": summary["Evidence_precision_mean"],
                "CI95": "-",
            },
            {
                "Metric": "Evidence Recall",
                "Mean": summary["Evidence_recall_mean"],
                "CI95": "-",
            },
            {
                "Metric": "Evidence F1",
                "Mean": summary["Evidence_f1_mean"],
                "CI95": "-",
            },
            {"Metric": "Faithfulness (token coverage)", "Mean": summary["Faithfulness_mean"], "CI95": "-"},
            {"Metric": "Attribution Accuracy", "Mean": summary["Attribution_accuracy_mean"], "CI95": "-"},
            {"Metric": "Hallucination Rate (lower is better)", "Mean": summary["Hallucination_rate_mean"], "CI95": "-"},
            {
                "Metric": "Latency (mean, s)",
                "Mean": summary["Latency_mean_s"],
                "CI95": "-",
            },
            {
                "Metric": "Latency (p95, s)",
                "Mean": summary["Latency_p95_s"],
                "CI95": "-",
            },
        ]
    )
    table_df.to_csv(outdir / "metrics_table.csv", index=False)

    # Plots (matplotlib only)
    # 1) Metric bar plot
    plt.figure()
    plt.bar(["HR@k", "RA", "GAA"], [summary["HR@k_mean"], summary["RA_mean"], summary["GAA_mean"]])
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(outdir / "metrics_bar.png", dpi=200)
    plt.close()

    # 2) Latency histogram
    plt.figure()
    plt.hist(df["latency_s"].values, bins=15)
    plt.xlabel("Latency (s)")
    plt.ylabel("Count")
    plt.title("Latency Distribution")
    plt.tight_layout()
    plt.savefig(outdir / "latency_hist.png", dpi=200)
    plt.close()

    print("✅ Done.")
    print("Outputs:")
    print(f" - {outdir / 'per_query.csv'}")
    print(f" - {outdir / 'metrics_table.csv'}")
    print(f" - {outdir / 'summary.json'}")
    print(f" - {outdir / 'metrics_bar.png'}")
    print(f" - {outdir / 'latency_hist.png'}")
    print(f" - {log_path}")


if __name__ == "__main__":
    main()

