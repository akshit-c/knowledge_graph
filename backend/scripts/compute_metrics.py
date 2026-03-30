import json
import statistics
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

INP = Path("backend/eval/results_100.jsonl")

y_true = []   # 1 = answerable, 0 = NOT_IN_MEMORY
y_pred = []   # 1 = answered, 0 = NOT_IN_MEMORY
scores = []   # confidence proxy (kg_score or similarity)
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

        # score proxy for ROC (use kg_score if exists)
        kg_score = 0.0
        try:
            if ex.get("sources"):
                kg_score = ex["sources"][0].get("kg_score", 0.0)
        except Exception:
            pass
        scores.append(kg_score)

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
    "latency_mean": lat_mean,
    "latency_p95": lat_p95
}, indent=2))

print("\nSaved metrics to", out)
print("ROC-AUC:", round(roc_auc, 3))
