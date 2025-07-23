# compare_training_results.py
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT_DIR = Path("/workflow/outputs")
OUT_FILE = OUT_DIR / "comparison"

HIGHER = {
    "roc_auc", "pr_auc", "f1_fraud", "precision_fraud", 
    "recall_fraud", "accuracy", "ks"
}
LOWER = {
    "log_loss", "brier", "ece", "fit_time_sec", 
    "predict_time_sec", "inf_ms_row", "model_size_kb"
}

def load_json_input(name: str) -> dict:
    """Load from /workflow/inputs/{name} if exists, else treat name as literal JSON string."""
    p = Path(f"/workflow/inputs/{name}")
    raw = p.read_text().strip() if p.exists() else name
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"Failed to parse input {name}: {e}")
        return {}

# Load both results safely
ada_blob = load_json_input("ada_results")
gnb_blob = load_json_input("gnb_results")

# Consolidate and flatten
consolidated = {"AdaBoost": ada_blob, "GaussianNB": gnb_blob}
df = pd.DataFrame.from_dict(consolidated, orient="index")
df.index.name = "model"
print('df', df)

# Clean up bad values (e.g., dicts/lists or nulls)
for col in df.columns:
    df[col] = df[col].apply(lambda v: v if pd.api.types.is_scalar(v) else np.nan)


# Print and save pretty string output
print("Consolidated:")
print(df)

OUT_DIR.mkdir(parents=True, exist_ok=True)
payload = df.reset_index().to_string()
print("Payload is here:")
print(payload)

# Save plain-text table
(Path("/workflow/outputs/consolidated")).write_text(payload)
