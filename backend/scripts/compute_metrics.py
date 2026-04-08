import json
import statistics
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    log_loss,
)

INP = Path("backend/eval/results_100.jsonl")

y_true = []   # 1 = answerable, 0 = NOT_IN_MEMORY
y_pred = []   # 1 = answered, 0 = NOT_IN_MEMORY
scores = []   # confidence proxy (kg_score or similarity)
gate_probs = []  # probability query is answerable (trace.gate_p_answerable)
latencies = []

HR_hits = []
GAA_hits = []

with INP.open("r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)

        is_answerable = 1 if ex["label"] == "POS" else 0
        answer = (ex.get("answer") or "").strip()

        predicted_answerable = 0 if answer == "NOT_IN_MEMORY" else 1

        y_true.append(is_answerable)
        y_pred.append(predicted_answerable)

        # score proxy for ROC (prefer calibrated gate prob when available)
        kg_score = 0.0
        try:
            if ex.get("sources"):
                kg_score = ex["sources"][0].get("kg_score", 0.0)
        except Exception:
            pass

        trace = ex.get("trace") or {}
        gp = trace.get("gate_p_answerable", None)
        if gp is not None:
            try:
                gp = float(gp)
            except Exception:
                gp = None

        gate_probs.append(gp)
        scores.append(gp if gp is not None else kg_score)

        latencies.append(ex.get("latency_s", 0.0))

        # HR@k: at least one source retrieved for answerable queries
        if is_answerable:
            HR_hits.append(1 if ex.get("sources") else 0)
            # GAA: answered + sources exist
            GAA_hits.append(
                1 if predicted_answerable == 1 and ex.get("sources") else 0
            )

# Metrics
RA = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0) / max(1, sum(1 for v in y_true if v == 0))
HR = statistics.mean(HR_hits) if HR_hits else 0.0
GAA = statistics.mean(GAA_hits) if GAA_hits else 0.0

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

lat_mean = statistics.mean(latencies)
lat_p95 = sorted(latencies)[int(0.95 * len(latencies))]

cm = confusion_matrix(y_true, y_pred)

def expected_calibration_error(
    y_true_list: list[int],
    y_prob_list: list[float],
    n_bins: int = 10,
) -> float:
    """
    ECE (expected calibration error) with uniform probability bins.
    ECE = sum_b (|B|/N) * |acc(B) - conf(B)|.
    """
    y_true_arr = np.asarray(y_true_list, dtype=float)
    y_prob_arr = np.asarray(y_prob_list, dtype=float)
    # Clamp away from exact 0/1 for numerical stability.
    y_prob_arr = np.clip(y_prob_arr, 1e-6, 1.0 - 1e-6)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true_arr)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob_arr >= lo) & (y_prob_arr <= hi)
        else:
            mask = (y_prob_arr >= lo) & (y_prob_arr < hi)
        if not np.any(mask):
            continue
        acc = float(y_true_arr[mask].mean())
        conf = float(y_prob_arr[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)

    return float(ece)


calib_y_true: list[int] = []
calib_y_prob: list[float] = []
for yt, gp in zip(y_true, gate_probs):
    if gp is None:
        continue
    calib_y_true.append(int(yt))
    calib_y_prob.append(float(gp))

ece_val: Optional[float] = None
log_loss_val: Optional[float] = None
brier_val: Optional[float] = None
if calib_y_prob:
    ece_val = expected_calibration_error(calib_y_true, calib_y_prob, n_bins=10)
    brier_val = float(brier_score_loss(calib_y_true, calib_y_prob))
    log_loss_val = float(log_loss(calib_y_true, calib_y_prob, labels=[0, 1]))

print("\n=== CORE METRICS ===")
print("RA (Refusal Accuracy):", round(RA, 3))
print("HR@k:", round(HR, 3))
print("GAA:", round(GAA, 3))

print("\n=== CLASSIFICATION ===")
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1:", round(f1, 3))
print("Confusion Matrix:\n", cm)

print("\n=== LATENCY ===")
print("Mean (s):", round(lat_mean, 3))
print("P95 (s):", round(lat_p95, 3))

# ROC + PR data (saved for plots)
fpr, tpr, _ = roc_curve(y_true, scores)
roc_auc = roc_auc_score(y_true, scores)

prec_curve, rec_curve, _ = precision_recall_curve(y_true, scores)

out = Path("backend/eval/metrics.json")
out.write_text(json.dumps({
    "RA": RA,
    "HR@k": HR,
    "GAA": GAA,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "brier_score": brier_val,
    "ece": ece_val,
    "log_loss": log_loss_val,
    "latency_mean": lat_mean,
    "latency_p95": lat_p95
}, indent=2))

print("\nSaved metrics to", out)
print("ROC-AUC:", round(roc_auc, 3))
