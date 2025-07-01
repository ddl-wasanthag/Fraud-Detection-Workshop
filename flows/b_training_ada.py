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

    train_fraud(model_obj, model_name, clean_filename, experiment_name)

if __name__ == "__main__":
    train_fraud_ada()
