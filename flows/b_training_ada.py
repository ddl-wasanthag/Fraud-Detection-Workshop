from pathlib import Path
import pandas as pd

from flows.generic_trainer import train_fraud
from helpers.domino_short_id import domino_short_id
from sklearn.ensemble import AdaBoostClassifier


def train_fraud_ada(clean_filepath, random_state=None):
    model_name = "AdaBoost"
    model_obj = AdaBoostClassifier(
        n_estimators=10,
        learning_rate=0.1,
        algorithm="SAMME",
        random_state=random_state
    )

    experiment_name = f"CC Fraud Classifier Training {domino_short_id()}"

    res = train_fraud(model_obj, model_name, clean_filepath, experiment_name)

    Path("/workflow/outputs/results_df").write_text(res)
    return "train_fraud_ada results"

preprocessed_df_path = Path("/workflow/inputs/preprocessed_df_path").read_text()

print('pp path', preprocessed_df_path)
import panadas as pd

try:
    preprocessed_df = pd.read_csv(preprocessed_df_path)
except Exception as e:
    print(f"Error reading preprocessed DataFrame: {e}")
    preprocessed_df = pd.DataFrame()

print('this', preprocessed_df.head(20))
res = train_fraud_ada(preprocessed_df_path)

# Write output
Path("/workflow/outputs/results_df").write_text(res)

