from flows.generic_trainer import train_fraud
from helpers.domino_short_id import domino_short_id
from xgboost import XGBClassifier

def train_fraud_xgb(random_state=None):
    model_obj = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=random_state
    )
    model_name = "XGBoost"
    clean_filename = 'preprocessing_processed_cc_transactions.csv'
    experiment_name = f"CC Fraud Classifier Training {domino_short_id()}"

    train_fraud(model_obj, model_name, clean_filename, experiment_name)


# train_fraud_xgb()

from pathlib import Path


preprocessed_df = Path("/workflow/inputs/preprocessed_df").read_text()

print(preprocessed_df)

# Write output
Path("/workflow/outputs/results_df").write_text(str(preprocessed_df))
