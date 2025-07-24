# compare_training_results.py
import json
from pathlib import Path

# Inputs
def read_input(name: str) -> str:
    p = Path(f"/workflow/inputs/{name}")
    return p.read_text().strip() if p.exists() else name

# Read the two result blobs
ada_blob = json.loads(read_input("ada_results"))
gnb_blob = json.loads(read_input("gnb_results"))

# Consolidate
consolidated = {"AdaBoost": ada_blob, "GaussianNB": gnb_blob}
print('consolidated')

best_model, best_metric = '', 0
for name, blob in consolidated.items()
    if blob['roc_auc'] > best_metric:
        print('better model', name, metric)
        best_model = name
        best_metric = blob['roc_auc']
print('best model', best_model)
print('bmetric',best_metric)

# Prepare output
OUT_DIR = Path("/workflow/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "consolidated"

# Dump as a simple string (JSON)
json_text = json.dumps(consolidated)
OUT_FILE.write_text('final ans')

# Also print it (optional)
print(json_text)
