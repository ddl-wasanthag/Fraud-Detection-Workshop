# File: trainer_gnb.py
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from exercises.d_TrainingAndEvaluation.generic_trainer import train_fraud
from exercises.1isjdofijwoiej

# Load DataFrame from dataset
try:
    transformed_df_filename = Path("/workflow/inputs/transformed_filename").read_text().strip()
except FileNotFoundError:
    transformed_df_filename = 'transformed_cc_transactions.csv'

model_name = 'GaussianNB'
model_obj = GaussianNB()

res = train_fraud(model_obj, model_name, transformed_df_filename)

print(f"Training {model_name} completed successfully")
print(res)

Path("/workflow/outputs/results").write_text(str(res))