# File: flows/b_training_ada_modified.py
import os
import io
import pandas as pd
from domino_data.datasets import DatasetClient, DatasetConfig
from exercises.b_DataProcessingAndExploration.preprocessing import run_data_ingestion_and_processing
from helpers.domino_short_id import domino_short_id


def save_to_dataset(df, filename, dataset_id="dataset-Fraud-Detection-Workshop-684ee0cf140dce3153f03833"):
    """Save DataFrame to Domino Dataset using REST API."""
    try:
        # Get token from environment or headers
        token = os.environ.get('DOMINO_USER_API_KEY', None)
        
        # Initialize dataset client
        dataset_client = DatasetClient(token=token)
        dataset = dataset_client.get_dataset(dataset_id)
        
        # Convert DataFrame to CSV bytes
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        # Upload to dataset
        print(f"📤 Uploading {filename} to dataset {dataset_id}")
        dataset.upload_file(filename, csv_buffer)
        
        print(f"✅ Successfully uploaded to dataset: {filename}")
        
        # List files to verify
        files = dataset.list_files()
        print(f"📁 Files in dataset: {[f['path'] for f in files]}")
        
        return filename
        
    except Exception as e:
        print(f"❌ Error with dataset upload: {e}")
        raise


def main():
    # Configuration
    raw_filename = "raw_cc_transactions.csv"
    clean_filename = "preprocessing_processed_cc_transactions.csv"
    experiment_name = f"CC Fraud Preprocessing {domino_short_id()}"
    dataset_id = "dataset-Fraud-Detection-Workshop-684ee0cf140dce3153f03833"

    # Process the data
    clean_df, clean_path, features_path, labels_path = run_data_ingestion_and_processing(
        raw_filename=raw_filename,
        clean_filename=clean_filename,
        experiment_name=experiment_name
    )
    
    print(f"\n🎉 Processing complete!")
    print(f"   DataFrame shape: {clean_df.shape}")
    
    # Save to dataset
    save_to_dataset(clean_df, clean_filename, dataset_id)
    
    # Also save the numpy features and labels if needed
    # Note: For numpy arrays, you'd need to convert them first
    import numpy as np
    
    # Load and save features
    features = np.load(features_path)
    features_df = pd.DataFrame(features)
    save_to_dataset(features_df, "preprocessing_features_processed.csv", dataset_id)
    
    # Load and save labels
    labels_df = pd.read_csv(labels_path)
    save_to_dataset(labels_df, "preprocessing_feature_labels.csv", dataset_id)
    
    # Write the filename to workflow output if in workflow mode
    if os.environ.get("DOMINO_IS_WORKFLOW_JOB", "false").lower() == "true":
        with open("/workflow/outputs/preprocessed_df_path", "w") as f:
            f.write(clean_filename)
        print(f"✅ Workflow output saved: {clean_filename}")
    
    return clean_filename


if __name__ == "__main__":
    main()