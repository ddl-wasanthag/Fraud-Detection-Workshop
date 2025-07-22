# File: exercises/d_TrainingAndEvaluation/compare_training_results.py
import json
import ast
from pathlib import Path
import pandas as pd
import numpy as np

OUT_DIR = Path("/workflow/outputs")
OUT_FILE = OUT_DIR / "comparison"   # must match outputs={'comparison': str}

HIGHER = {
    "roc_auc", "pr_auc",
    "f1_fraud", "precision_fraud", "recall_fraud",
    "accuracy", "ks"
}
LOWER = {
    "log_loss", "brier", "ece",
    "fit_time_sec", "predict_time_sec", "inf_ms_row",
    "model_size_kb"
}

def _read_input(name: str) -> str:
    p = Path(f"/workflow/inputs/{name}")
    return p.read_text().strip() if p.exists() else name

def _to_dict(blob: str):
    if isinstance(blob, str):
        p = Path(blob)
        if p.exists():
            return json.loads(p.read_text())
    try:
        return json.loads(blob)
    except Exception:
        return ast.literal_eval(blob)

def main():
    ada_blob = _to_dict(_read_input("ada_results"))
    gnb_blob = _to_dict(_read_input("gnb_results"))
    
    consolidated = {"AdaBoost": ada_blob, "GaussianNB": gnb_blob}
    print("consolidated", consolidated)

    # Build DF
    df = pd.DataFrame.from_dict(consolidated, orient="index")
    df.index.name = "model"

    # Flatten non-scalars -> NaN, then coerce numerics
    df = df.applymap(lambda x: x if np.isscalar(x) else np.nan)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") if df[c].dtype == object else df[c]

    print('df', df)

    # Rank numerics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    ranks = {}
    for col in numeric_cols:
        ascending = col in LOWER
        ranks[f"{col}_rank"] = df[col].rank(ascending=ascending, method="min")
    if ranks:
        df = pd.concat([df, pd.DataFrame(ranks, index=df.index)], axis=1)

    # Write required output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_text = df.reset_index().to_csv(index=False)
    OUT_FILE.write_text(csv_text)

    print("\n=== Comparison CSV ===\n")
    print(csv_text)

if __name__ == "__main__":
    main()
