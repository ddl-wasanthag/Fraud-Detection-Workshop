from flows.generic_trainer import train_fraud
from helpers.domino_short_id import domino_short_id
from sklearn.naive_bayes import GaussianNB

def train_fraud_gnb(random_state=None):
    model_obj = GaussianNB()
    model_name = "NaiveBayes"
    clean_filename = 'preprocessing_processed_cc_transactions.csv'
    experiment_name = f"CC Fraud Classifier Training {domino_short_id()}"

    train_fraud(model_obj, model_name, clean_filename, experiment_name)

# train_fraud_gnb()

from pathlib import Path


preprocessed_df = Path("/workflow/inputs/preprocessed_df").read_text()

print(preprocessed_df)

# Write output
Path("/workflow/outputs/results_df").write_text(str(preprocessed_df))
