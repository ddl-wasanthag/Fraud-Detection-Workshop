# File: compare_training_results.py
import json
import ast
from pathlib import Path

import pandas as pd


def _read_workflow_input(name: str) -> str:
    """
    Read a Flyte/Domino workflow input file. If it doesn't exist, just return the raw name.
    """
    p = Path(f"/workflow/inputs/{name}")
    if p.exists():
        return p.read_text().strip()
    return name


def _safe_to_dict(blob: str):
    print('safe to dict', blob)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return ast.literal_eval(blob)


def main():
    # ----- 1) Load the consolidated JSON -----
    ada_blob = _safe_to_dict(_read_workflow_input("ada_results"))
    gnb_blob = _safe_to_dict(_read_workflow_input("gnb_results"))
    consolidated = {"AdaBoost": ada_blob, "GaussianNB": gnb_blob}
    
    print(consolidated)

    print('ada', ada_blob)
    print('gnb', gnb_blob)
    
    # consolidated = _safe_to_dict(raw)  # dict: {model_name: {metric: value, ...}}

    # ----- 2) Build a DataFrame -----
    df = pd.DataFrame(consolidated).T  # rows = models
    df.index.name = "model"
    df.reset_index(inplace=True)

    # ----- 3) Basic comparisons -----
    # Figure out which columns are numeric metrics
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Rank (1 = best) for metrics where higher is better
    higher_better = {"roc_auc", "average_precision", "f1_score", "accuracy", "balanced_accuracy"}
    # Everything else numeric but not in higher_better assume lower-is-better (log_loss, brier, etc.)
    lower_better = set(numeric_cols) - higher_better

    rank_df = pd.DataFrame(index=df["model"])
    for col in numeric_cols:
        if col in higher_better:
            rank_df[col] = df[col].rank(ascending=False, method="min")
        else:
            rank_df[col] = df[col].rank(ascending=True, method="min")

    # Build a quick "winner per metric" dict
    winners = {}
    for col in numeric_cols:
        best_row = rank_df[col].idxmin()  # index is model name
        winners[col] = best_row

    # ----- 4) Write outputs -----
    # Primary output: a CSV table combining metrics + ranks
    out_df = df.set_index("model").join(rank_df.add_suffix("_rank"))
    csv_text = out_df.to_csv()

    # Secondary: JSON summary of winners
    summary_json = json.dumps({"winners": winners}, indent=2)

    # Domino/Flyte capture
    out_dir = Path("/workflow/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "comparison").write_text(csv_text)
    (out_dir / "summary").write_text(summary_json)

    # Also print to stdout so it shows in logs
    print("\n=== Model Comparison Table ===\n")
    print(out_df)
    print("\n=== Winners by Metric ===\n")
    print(summary_json)


if __name__ == "__main__":
    main()
