import os
from pathlib import Path
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from domino import Domino

from flows.generic_trainer import train_fraud
from helpers.domino_short_id import domino_short_id


def get_domino_client():
    """Initialize Domino client with proper authentication."""
    return Domino(
        project=os.environ.get("DOMINO_PROJECT_NAME", "default-project"),
        api_key=os.environ.get("DOMINO_USER_API_KEY"),
        host=os.environ.get("DOMINO_API_HOST", "https://trial.dominodatalab.com")
    )


def download_from_dataset(dataset_name, filename, local_path):
    """Download a file from a Domino Dataset."""
    domino = get_domino_client()
    
    # Get the dataset ID
    datasets = domino.datasets_list()
    dataset_id = None
    for dataset in datasets:
        if dataset['name'] == dataset_name:
            dataset_id = dataset['id']
            break
    
    if dataset_id is None:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    
    # Download the file
    print(f"Downloading {filename} from dataset {dataset_name}")
    domino.datasets_download_files(
        dataset_id=dataset_id,
        file_path=filename,
        local_path=local_path
    )
    
    return local_path


def train_fraud_ada(clean_filepath, random_state=None):
    """Train AdaBoost classifier for fraud detection."""
    model_obj = AdaBoostClassifier(
        n_estimators=10,
        learning_rate=0.1,
        algorithm="SAMME",
        random_state=random_state
    )
    
    model_name = "AdaBoost"
    experiment_name = f"CC Fraud Classifier Training {domino_short_id()}"
    
    return train_fraud(model_obj, model_name, clean_filepath, experiment_name)


def main():
    # Read the preprocessed data path from workflow input
    preprocessed_df_path = Path("/workflow/inputs/preprocessed_df_path").read_text().strip()
    print(f'📍 Processing data from: {preprocessed_df_path}')
    
    # Check if it's a dataset path or local file path
    if preprocessed_df_path.startswith("/domino/datasets/"):
        print("🔍 Detected dataset path, checking if file exists...")
        
        if os.path.exists(preprocessed_df_path):
            print("✅ Dataset file found at mount path")
            final_path = preprocessed_df_path
        else:
            print("❌ Dataset file not found at mount path, downloading...")
            # Extract dataset name and filename from path
            path_parts = preprocessed_df_path.split('/')
            dataset_name = path_parts[3]  # /domino/datasets/DATASET_NAME/filename
            filename = path_parts[4]
            
            # Download to local temp file
            local_path = f"/tmp/{filename}"
            download_from_dataset(dataset_name, filename, local_path)
            final_path = local_path
    else:
        # It's a local file path
        final_path = preprocessed_df_path
    
    # Verify the file exists and is readable
    if not os.path.exists(final_path):
        raise FileNotFoundError(f"Data file not found: {final_path}")
    
    try:
        df = pd.read_csv(final_path)
        print(f'✅ Successfully loaded {len(df)} rows, {len(df.columns)} columns')
        print(f'Columns: {list(df.columns)}')
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        raise
    
    # Train the model
    try:
        result = train_fraud_ada(final_path)
        print(f"✅ Training completed successfully")
        
        # Write output
        Path("/workflow/outputs/results_df").write_text(result)
        return result
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()