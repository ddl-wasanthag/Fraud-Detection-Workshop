# File: flows/b_training_ada.py
import yaml
from sklearn.ensemble import AdaBoostClassifier
from flows.generic_trainer import train_fraud
from helpers.domino_short_id import domino_short_id
from helpers.dataset_utils import load_from_data_source


def main(preprocessed_df_path, random_state=None):
    # Read the filename from workflow input
    print('📍 Starting AdaBoost training workflow')
    print('preprocessed_df_filepath', preprocessed_df_path)
    with open("/workflow/inputs/preprocessed_df_path", "r") as f:
        preprocessed_df_filepath = f.read().strip()
    print('preprocessed_df_filepathafter', preprocessed_df_filepath)

    print(f'📍 Processing data from datasource: {preprocessed_df_filepath}')
    
    # Load DataFrame from data source
    preprocessed_df = load_from_data_source(preprocessed_df_filepath)
    print(f'Columns: {list(preprocessed_df.columns)}')
    
    # Train the model
    model_obj = AdaBoostClassifier(
        n_estimators=10,
        learning_rate=0.1,
        algorithm="SAMME",
    )
    model_name = "AdaBoost"
    experiment_name = f"CC Fraud Classifier Training {domino_short_id()}"
    
    result =  train_fraud(model_obj, model_name, preprocessed_df, experiment_name, preprocessed_df_filepath, random_state=random_state)
    print(f"✅ Training completed successfully")
    
    # Write output
    with open("/workflow/outputs/results_df", "w") as f:
        if isinstance(result, dict):
            f.write(yaml.dump(result, default_flow_style=False))
        else:
            f.write(str(result))
    
    return result


if __name__ == "__main__":

    print('📍MAIN Starting AdaBoost training workflow')
    with open("/workflow/inputs/preprocessed_df_path", "r") as f:
        preprocessed_df_filepath = f.read().strip()
    print('mAININ', preprocessed_df_filepath)
    print('MAINpreprocessed_df_filepath', preprocessed_df_filepath)
    main(preprocessed_df_filepath)
