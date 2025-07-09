# File: trainer_ada.py
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier
from exercises.d_TrainingAndEvaluation.generic_trainer import train_fraud

# Load DataFrame from dataset
try:
    transformed_df_filename = Path("/workflow/inputs/transformed_filename").read_text().strip()
except FileNotFoundError:
    transformed_df_filename = 'transformed_cc_transactions.csv'

model_name = 'AdaBoost'
model_obj = AdaBoostClassifier(
            n_estimators=10,
            learning_rate=0.1,
            algorithm="SAMME",
        )

res = train_fraud(model_obj, model_name, transformed_df_filename)

print(f"Training {model_name} completed successfully")
print(res)

Path("/workflow/outputs/results").write_text(str(res))