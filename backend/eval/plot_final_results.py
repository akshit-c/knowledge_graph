import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.linear_model import LinearRegression

# -----------------------------
# Paths
# -----------------------------
RESULTS_PATH = Path("backend/eval/results_100.jsonl")
METRICS_PATH = Path("backend/eval/metrics.json")
OUT_DIR = Path("backend/eval/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid", font_scale=1.2)

# -----------------------------
# Load data
# -----------------------------
records = []
with open(RESULTS_PATH) as f:
    for line in f:
        records.append(json.loads(line))

with open(METRICS_PATH) as f:
    metrics = json.load(f)

# Extract labels and probabilities
y_true = []
y_score = []
latencies = []

for r in records:
    y_true.append(1 if r["label"] == "positive" else 0)
    y_score.append(r["trace"]["gate_p_answerable"])
    latencies.append(r.get("latency_s", 0.0))

y_true = np.array(y_true)
y_score = np.array(y_score)
latencies = np.array(latencies)

# -----------------------------
# 1️⃣ ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Calibrated Refusal Gate)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "roc_curve.png", dpi=300)
plt.close()

# -----------------------------
# 2️⃣ Precision–Recall Curve
# -----------------------------
precision, recall, _ = precision_recall_curve(y_true, y_score)

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.tight_layout()
plt.savefig(OUT_DIR / "pr_curve.png", dpi=300)
plt.close()

# -----------------------------
# 3️⃣ Confusion Matrix (Stage 2B)
# -----------------------------
threshold = metrics["threshold_fp1"]
y_pred = (y_score >= threshold).astype(int)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Refuse", "Answer"],
    yticklabels=["Not Answerable", "Answerable"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Balanced Mode)")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=300)
plt.close()

# -----------------------------
# 4️⃣ Decision Score vs Truth (Scatter + Regression)
# -----------------------------
X = y_score.reshape(-1, 1)
y = y_true

model = LinearRegression().fit(X, y)
y_line = model.predict(X)

plt.figure(figsize=(6, 4))
plt.scatter(y_score, y_true, alpha=0.6, label="Queries")
plt.plot(y_score, y_line, color="red", linewidth=2, label="Regression")
plt.xlabel("Gate Probability (Answerable)")
plt.ylabel("Ground Truth")
plt.title("Decision Score vs Ground Truth")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "decision_scatter.png", dpi=300)
plt.close()

# -----------------------------
# 5️⃣ Latency Distribution (Violin)
# -----------------------------
plt.figure(figsize=(6, 4))
sns.violinplot(y=latencies, inner="quartile")
plt.ylabel("Latency (seconds)")
plt.title("Latency Distribution")
plt.tight_layout()
plt.savefig(OUT_DIR / "latency_violin.png", dpi=300)
plt.close()

# -----------------------------
# 6️⃣ Metric Comparison Bar Chart
# -----------------------------
labels = ["Baseline", "Hybrid", "Calibrated"]
recall_vals = [0.383, 0.583, 0.483]
ra_vals = [1.0, 0.925, 0.975]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(7, 4))
plt.bar(x - width / 2, recall_vals, width, label="Recall")
plt.bar(x + width / 2, ra_vals, width, label="Refusal Accuracy")
plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Recall vs Safety Trade-off")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "recall_vs_ra.png", dpi=300)
plt.close()

print("✅ All publication-grade figures saved to:", OUT_DIR)
