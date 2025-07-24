# compare_training_results.py
import json
from pathlib import Path

# Which metrics are better when higher vs lower
HIGHER = {
    "roc_auc", "pr_auc", "f1_fraud", "precision_fraud",
    "recall_fraud", "accuracy", "ks"
}
LOWER = {
    "log_loss", "brier", "ece", "fit_time_sec",
    "predict_time_sec", "inf_ms_row", "model_size_kb"
}

def read_input(name: str) -> str:
    p = Path(f"/workflow/inputs/{name}")
    return p.read_text().strip() if p.exists() else name

# Load blobs
ada = json.loads(read_input("ada_results"))
gnb = json.loads(read_input("gnb_results"))
consolidated = {"AdaBoost": ada, "GaussianNB": gnb}

# Score each model
models = list(consolidated.keys())
scores = {m: 0 for m in models}

for metric in HIGHER:
    if metric in ada and metric in gnb:
        if ada[metric] > gnb[metric]:
            scores["AdaBoost"] += 1
        elif gnb[metric] > ada[metric]:
            scores["GaussianNB"] += 1

for metric in LOWER:
    if metric in ada and metric in gnb:
        if ada[metric] < gnb[metric]:
            scores["AdaBoost"] += 1
        elif gnb[metric] < ada[metric]:
            scores["GaussianNB"] += 1

# Determine best
best = max(scores, key=scores.get)
message = f"Best model based on metrics comparison: {best}"

# Write output
out_dir = Path("/workflow/outputs")
out_dir.mkdir(parents=True, exist_ok=True)
(Path("/workflow/outputs/best_model.txt")).write_text(message)

print(message)
