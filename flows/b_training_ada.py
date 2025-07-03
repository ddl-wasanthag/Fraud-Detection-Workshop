# File: flows/b_training_ada.py
import os
import io
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from domino_data.data_sources import DataSourceClient
from flows.generic_trainer import train_fraud
from helpers.domino_short_id import domino_short_id


def load_from_data_source(filename):
    """Load DataFrame from Domino Data Source."""
    try:
        # Get the data source client
        ds_client = DataSourceClient()
        ds = ds_client.get_datasource("credit_card_fraud_detection")
        
        # Download the file
        print(f"📥 Downloading {filename} from data source")
        buf = io.BytesIO()
        ds.download_fileobj(filename, buf)
        buf.seek(0)
        
        # Load as DataFrame
        df = pd.read_csv(buf)
        print(f"✅ Loaded DataFrame with {len(df)} rows, {len(df.columns)} columns")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading from data source: {e}")
        raise


def create_temp_csv_for_training(df, filename="/tmp/temp_training_data.csv"):
    """Create temporary CSV file for training function."""
    df.to_csv(filename, index=False)
    return filename


def train_fraud_ada(df, random_state=None):
    """Train AdaBoost classifier from DataFrame."""
    # Create temp file for train_fraud function
    temp_path = create_temp_csv_for_training(df)
    
    model_obj = AdaBoostClassifier(
        n_estimators=10,
        learning_rate=0.1,
        algorithm="SAMME",
        random_state=random_state
    )
    
    model_name = "AdaBoost"
    experiment_name = f"CC Fraud Classifier Training {domino_short_id()}"
    
    return train_fraud(model_obj, model_name, temp_path, experiment_name)


def main():
    # Read the filename from workflow input
    with open("/workflow/inputs/preprocessed_df_path", "r") as f:
        filename = f.read().strip()
    
    print(f'📍 Processing data from file: {filename}')
    
    # Load DataFrame from data source
    df = load_from_data_source(filename)
    print(f'Columns: {list(df.columns)}')
    
    # Train the model
    result = train_fraud_ada(df)
    print(f"✅ Training completed successfully")
    
    # Write output
    with open("/workflow/outputs/results_df", "w") as f:
        f.write(result)
    
    return result


if __name__ == "__main__":
    main()
