
# File: flows/a_preprocessing.py
import os
import pandas as pd
from domino_data.datasets import DatasetClient
from exercises.b_DataProcessingAndExploration.preprocessing import run_data_ingestion_and_processing
from helpers.domino_short_id import domino_short_id


def save_to_domino_dataset(df, filename, dataset_name="credit_card_fraud_detection"):
    """Save DataFrame to Domino Dataset."""
    try:
        # Use DatasetClient instead of DataSourceClient
        dataset_client = DatasetClient()
        dataset = dataset_client.get_dataset(dataset_name)
        
        print(f"📤 Uploading {filename} to dataset '{dataset_name}'")
        print(f"   DataFrame shape: {df.shape}")
        
        # Save DataFrame directly to dataset
        dataset.upload_dataframe(df, filename)
        
        print(f"✅ Successfully uploaded to dataset: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Failed to upload to dataset: {e}")
        print(f"   Error type: {type(e).__name__}")
        raise

def verify_dataset_upload(filename, dataset_name="credit_card_fraud_detection"):
    """Verify the file exists in the dataset."""
    try:
        dataset_client = DatasetClient()
        dataset = dataset_client.get_dataset(dataset_name)
        
        # List files in dataset
        files = dataset.list_files()
        if filename in files:
            print(f"✅ Verified: {filename} exists in dataset")
            return True
        else:
            print(f"❌ File not found in dataset: {filename}")
            print(f"   Available files: {files}")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    """Main preprocessing function with dataset upload."""
    
    # Configuration
    raw_filename = "raw_cc_transactions.csv"
    clean_filename = "preprocessing_processed_cc_transactions.csv"
    dataset_name = "credit_card_fraud_detection"
    experiment_name = f"CC Fraud Preprocessing {domino_short_id()}"

    # Process the data
    clean_df, _, _, _ = run_data_ingestion_and_processing(
        raw_filename=raw_filename,
        clean_filename=clean_filename,
        experiment_name=experiment_name
    )
    
    print(f"\n🎉 Processing complete!")
    print(f"   DataFrame shape: {clean_df.shape}")
    
    # Upload to dataset
    try:
        uploaded_filename = save_to_domino_dataset(clean_df, clean_filename, dataset_name)
        
        # Verify upload
        if verify_dataset_upload(uploaded_filename, dataset_name):
            print("✅ Dataset upload verified")
        else:
            print("⚠️  Trying alternative upload method...")
            uploaded_filename = save_to_domino_dataset_alternative(clean_df, clean_filename, dataset_name)
            
    except Exception as e:
        print(f"❌ Dataset upload failed: {e}")
        # Fallback to local save
        local_path = f"/tmp/{clean_filename}"
        clean_df.to_csv(local_path, index=False)
        uploaded_filename = local_path
        print(f"💾 Saved locally as fallback: {uploaded_filename}")
    
    # Write output for workflow
    output_path = "/workflow/outputs/preprocessed_df_path"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(uploaded_filename)
    
    print(f"✅ Workflow output saved: {uploaded_filename}")
    return uploaded_filename


if __name__ == "__main__":
    main()