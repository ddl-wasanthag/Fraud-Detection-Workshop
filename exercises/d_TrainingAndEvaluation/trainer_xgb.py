# File: trainer_xgb.py
from pathlib import Path
from xgboost import XGBClassifier
from exercises.d_TrainingAndEvaluation.generic_trainer import train_fraud

# Load DataFrame from dataset
try:
    transformed_df_filename = Path("/workflow/inputs/transformed_filename").read_text().strip()
except FileNotFoundError:
    transformed_df_filename = 'transformed_cc_transactions.csv'

model_name = 'XGBoost'
model_obj = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="auc",
        )

res = train_fraud(model_obj, model_name, transformed_df_filename)

print(f"Training {model_name} completed successfully")
print(res)

workflow_output_path = Path("/workflow/outputs/results")
if workflow_output_path.parent.exists():
    workflow_output_path.write_text(str(res))

