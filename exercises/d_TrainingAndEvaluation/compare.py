# compare_training_results.py
import json
from pathlib import Path


def read_input(name: str) -> str:
    p = Path(f"/workflow/inputs/{name}")
    return p.read_text().strip() if p.exists() else name

# Load blobs
ada = json.loads(read_input("ada_results"))
gnb = json.loads(read_input("gnb_results"))
consolidated = {"AdaBoost": ada, "GaussianNB": gnb}

best_metric = 0
best_model = 'none'
for name, res in consolidated.items()
    if res['roc_auc'] > best_metric:
        best_model = name
        best_metric = res['roc_auc']

message = "Best model based on metrics comparison: " + best_model

# Write output
out_dir = Path("/workflow/outputs")
out_dir.mkdir(parents=True, exist_ok=True)
(Path("/workflow/outputs/best_model.txt")).write_text(message)

print(message)
