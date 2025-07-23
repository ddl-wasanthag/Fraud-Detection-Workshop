import json, ast
from pathlib import Path
import pandas as pd
import numpy as np

OUT_DIR  = Path("/workflow/outputs")
OUT_FILE = OUT_DIR / "comparison"

HIGHER = {
    "roc_auc", "pr_auc",
    "f1_fraud", "precision_fraud", "recall_fraud",
    "accuracy", "ks"
}
LOWER  = {
    "log_loss", "brier", "ece",
    "fit_time_sec", "predict_time_sec", "inf_ms_row",
    "model_size_kb"
}

def read_input(name: str) -> str:
    p = Path(f"/workflow/inputs/{name}")
    return p.read_text().strip() if p.exists() else name

def to_dict(blob: str):
    if isinstance(blob, str):
        p = Path(blob)
        if p.exists():
            return json.loads(p.read_text())
    return json.loads(blob)

def flatten_scalars(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = df[c].apply(lambda v: v if np.isscalar(v) else np.nan)
    return df

ada_blob = to_dict(read_input("ada_results"))
gnb_blob = to_dict(read_input("gnb_results"))
consolidated = {"AdaBoost": ada_blob, "GaussianNB": gnb_blob}
print("consolidated", consolidated)

df = pd.DataFrame.from_dict(consolidated, orient="index")
df.index.name = "model"
df = flatten_scalars(df)
print('flattened df')

# numeric coercion
for c in df.columns:
    if df[c].dtype == object:
        df[c] = pd.to_numeric(df[c], errors="ignore")

# rank only numeric cols
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
rank_cols = {}
for col in numeric_cols:
    if col in HIGHER:
        asc = False
    elif col in LOWER:
        asc = True
    else:
        # heuristic: treat 'loss','time','err','ms' as lower-better
        asc = any(k in col.lower() for k in ("loss", "time", "err", "ms"))
    rank_cols[f"{col}_rank"] = df[col].rank(ascending=asc, method="min")

if rank_cols:
    df = pd.concat([df, pd.DataFrame(rank_cols, index=df.index)], axis=1)

print('rank cols', rank_cols)
print('numeric cols', numeric_cols)
print('df', df)
print('df cols', df.columns)

OUT_DIR.mkdir(parents=True, exist_ok=True)
# Convert NaN to None for JSON serialization
payload = df.reset_index().fillna(None).to_dict(orient="records")
print('df df df', payload)

OUT_FILE.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
print(f"[compare] wrote {OUT_FILE} ({OUT_FILE.stat().st_size} bytes)")
print(f"[compare] sample:\n{json.dumps(payload[:1], indent=2)}")