from pathlib import Path
import pandas as pd

preprocessed_df_path = Path("/workflow/inputs/preprocessed_df_path").read_text()

print(preprocessed_df_path)

test_df = pd.DataFrame({
    "transaction_id": [1, 2, 3],
    "amount": [100.0, 200.0, 300.0],
    "is_fraud": [0, 0, 1]
})

# Write output
Path("/workflow/outputs/results_df").write_text(str(test_df))

from flows.generic_trainer import train_fraud
from helpers.domino_short_id import domino_short_id
from sklearn.ensemble import AdaBoostClassifier

def train_fraud_ada(random_state=None):
    model_obj = AdaBoostClassifier(
        n_estimators=10,
        learning_rate=0.1,
        algorithm="SAMME",
        random_state=random_state
    )

    model_name = "AdaBoost"
    clean_filename = 'preprocessing_processed_cc_transactions.csv'
    experiment_name = f"CC Fraud Classifier Training {domino_short_id()}"

    res = train_fraud(model_obj, model_name, clean_filename, experiment_name)

    Path("/workflow/outputs/results_df").write_text(res)
    return "train_fraud_ada results"

if __name__ == "__main__":
    train_fraud_ada()
