# File: flows/b_training_gnb.py
import yaml
from sklearn.naive_bayes import GaussianNB
from flows.generic_trainer import train_fraud
from helpers.domino_short_id import domino_short_id
from helpers.dataset_utils import load_from_data_source

def main():
    # Read the filename from workflow input
    with open("/workflow/inputs/preprocessed_df_path", "r") as f:
        preprocessed_df_filepath = f.read().strip()
    
    print(f'📍 Processing data from datasource: {preprocessed_df_filepath}')
    
    # Load DataFrame from data source
    preprocessed_df = load_from_data_source(preprocessed_df_filepath)
    print(f'Columns: {list(preprocessed_df.columns)}')
    
    # Train the model
    model_obj = GaussianNB()
    model_name = "NaiveBayes"
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
    main()
