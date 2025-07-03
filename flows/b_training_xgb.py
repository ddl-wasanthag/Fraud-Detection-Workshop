# File: flows/b_training_xgb.py
import yaml
from xgboost import XGBClassifier
from flows.generic_trainer import train_fraud
from helpers.domino_short_id import domino_short_id
from helpers.dataset_utils import load_from_data_source


def main(preprocessed_df_path, random_state=None):
    # Read the filename from workflow input
    print('📍 Starting AdaBoost training workflow')

    print(f'📍 Processing data from datasource: {preprocessed_df_path}')
    
    # Load DataFrame from data source
    preprocessed_df = load_from_data_source(preprocessed_df_path)
    print(f'Columns: {list(preprocessed_df.columns)}')
    
    # Train the model
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
    experiment_name = f"CC Fraud Classifier Training {domino_short_id()}"
    
    result =  train_fraud(model_obj, model_name, preprocessed_df, experiment_name, preprocessed_df_path, random_state=random_state)
    print(f"✅ Training completed successfully")
    
    # Write output
    with open("/workflow/outputs/results_df", "w") as f:
        if isinstance(result, dict):
            f.write(yaml.dump(result, default_flow_style=False))
        else:
            f.write(str(result))
    
    return result


if __name__ == "__main__":

    print('Starting AdaBoost training workflow')
    with open("/workflow/inputs/preprocessed_df_path", "r") as f:
        preprocessed_df_path = f.read().strip()
    main(preprocessed_df_path)
