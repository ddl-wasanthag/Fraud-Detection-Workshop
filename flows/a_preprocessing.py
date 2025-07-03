import os
import io
import pandas as pd
from domino_data.datasets import DatasetClient
from pathlib import Path


def save_to_domino_dataset(df, filename, dataset_name="credit_card_fraud_detection"):
    """Save DataFrame to Domino Dataset using proper API."""
    try:
        # Get dataset client and dataset
        client = DatasetClient()
        dataset = client.get_dataset(dataset_name)
        
        print(f"📤 Uploading {filename} to dataset '{dataset_name}'")
        print(f"   DataFrame shape: {df.shape}")
        
        # Convert DataFrame to CSV bytes
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        # Upload using the dataset's upload method
        dataset.upload_fileobj(filename, csv_buffer)
        
        print(f"✅ Successfully uploaded to dataset: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Failed to upload to dataset: {e}")
        print(f"   Error type: {type(e).__name__}")
        raise


def save_to_domino_dataset_alternative(df, filename, dataset_name="credit_card_fraud_detection"):
    """Alternative method using temporary file upload."""
    try:
        # Save to temp file
        temp_path = f"/tmp/{filename}"
        df.to_csv(temp_path, index=False)
        
        # Get dataset
        client = DatasetClient()
        dataset = client.get_dataset(dataset_name)
        
        # Upload file
        dataset.upload_file(temp_path, filename)
        
        # Clean up
        os.remove(temp_path)
        
        print(f"✅ Successfully uploaded via temp file: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Alternative upload failed: {e}")
        raise


def verify_dataset_upload(filename, dataset_name="credit_card_fraud_detection"):
    """Verify the file exists in the dataset."""
    try:
        client = DatasetClient()
        dataset = client.get_dataset(dataset_name)
        
        # List files using the client method (not dataset method)
        files = client.list_files(dataset_name, "", 1000)
        
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


def list_dataset_files(dataset_name="credit_card_fraud_detection", prefix="", limit=1000):
    """List files in the dataset."""
    try:
        client = DatasetClient()
        files = client.list_files(dataset_name, prefix, limit)
        print(f"📁 Files in dataset '{dataset_name}': {files}")
        return files
    except Exception as e:
        print(f"❌ Failed to list files: {e}")
        return []


def main():
    """Main preprocessing function with dataset upload."""
    from exercises.b_DataProcessingAndExploration.preprocessing import run_data_ingestion_and_processing
    from helpers.domino_short_id import domino_short_id
    
    # Configuration
    raw_filename = "raw_cc_transactions.csv"
    clean_filename = "preprocessing_processed_cc_transactions.csv"
    dataset_name = "credit_card_fraud_detection"
    experiment_name = f"CC Fraud Preprocessing {domino_short_id()}"

    # Check existing files first
    print(f"📋 Checking existing files in dataset...")
    list_dataset_files(dataset_name)

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
            verify_dataset_upload(uploaded_filename, dataset_name)
            
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
    
    # Show final file list
    print(f"\n📋 Final files in dataset:")
    list_dataset_files(dataset_name)
    
    return uploaded_filename


if __name__ == "__main__":
    main()