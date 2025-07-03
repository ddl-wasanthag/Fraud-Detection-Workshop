import os
import pandas as pd
from dominodatalab import Domino  # This is the correct import for python-domino
from pathlib import Path


def save_to_domino_dataset(df, filename, dataset_name="credit_card_fraud_detection"):
    """Save DataFrame to Domino Dataset using python-domino library."""
    try:
        # Initialize Domino client
        domino = Domino()
        
        print(f"📤 Uploading {filename} to dataset '{dataset_name}'")
        print(f"   DataFrame shape: {df.shape}")
        
        # Save DataFrame to temporary file
        temp_path = f"/tmp/{filename}"
        df.to_csv(temp_path, index=False)
        
        # Upload to dataset using python-domino
        domino.datasets_upload(dataset_name, temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        print(f"✅ Successfully uploaded to dataset: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Failed to upload to dataset: {e}")
        print(f"   Error type: {type(e).__name__}")
        raise


def save_to_domino_dataset_alternative(df, filename, dataset_id):
    """Alternative method using dataset ID instead of name."""
    try:
        domino = Domino()
        
        # Save to temp file
        temp_path = f"/tmp/{filename}"
        df.to_csv(temp_path, index=False)
        
        # Upload using dataset ID
        domino.datasets_upload(dataset_id, temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        print(f"✅ Successfully uploaded via dataset ID: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Alternative upload failed: {e}")
        raise


def list_datasets():
    """List all datasets to find the correct name/ID."""
    try:
        domino = Domino()
        datasets = domino.datasets_list()
        print(f"📋 Available datasets: {datasets}")
        return datasets
    except Exception as e:
        print(f"❌ Failed to list datasets: {e}")
        return []


def find_dataset_id(dataset_name):
    """Find dataset ID by name."""
    try:
        domino = Domino()
        datasets = domino.datasets_list()
        
        for dataset in datasets:
            if dataset.get('name') == dataset_name:
                return dataset.get('id')
        
        print(f"❌ Dataset '{dataset_name}' not found")
        return None
        
    except Exception as e:
        print(f"❌ Error finding dataset: {e}")
        return None


def create_dataset_if_not_exists(dataset_name):
    """Create dataset if it doesn't exist."""
    try:
        domino = Domino()
        
        # Check if dataset exists
        dataset_id = find_dataset_id(dataset_name)
        if dataset_id:
            print(f"📁 Dataset '{dataset_name}' already exists (ID: {dataset_id})")
            return dataset_id
        
        # Create new dataset
        print(f"🆕 Creating dataset '{dataset_name}'")
        result = domino.datasets_create(dataset_name)
        
        # Extract ID from result
        dataset_id = result.get('id') if isinstance(result, dict) else result
        print(f"✅ Created dataset '{dataset_name}' with ID: {dataset_id}")
        return dataset_id
        
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return None


def main():
    """Main preprocessing function with dataset upload."""
    from exercises.b_DataProcessingAndExploration.preprocessing import run_data_ingestion_and_processing
    from helpers.domino_short_id import domino_short_id
    
    # Configuration
    raw_filename = "raw_cc_transactions.csv"
    clean_filename = "preprocessing_processed_cc_transactions.csv"
    dataset_name = "credit_card_fraud_detection"
    experiment_name = f"CC Fraud Preprocessing {domino_short_id()}"

    # List available datasets first
    print(f"📋 Checking available datasets...")
    list_datasets()
    
    # Find or create dataset
    dataset_id = create_dataset_if_not_exists(dataset_name)
    if not dataset_id:
        print("❌ Could not create or find dataset")
        return None

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
        # Try with dataset name first
        uploaded_filename = save_to_domino_dataset(clean_df, clean_filename, dataset_name)
        
    except Exception as e:
        print(f"❌ Upload with name failed: {e}")
        try:
            # Try with dataset ID
            uploaded_filename = save_to_domino_dataset_alternative(clean_df, clean_filename, dataset_id)
        except Exception as e2:
            print(f"❌ Upload with ID also failed: {e2}")
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
    mai