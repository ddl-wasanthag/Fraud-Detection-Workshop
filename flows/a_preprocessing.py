# from pathlib import Path
# from exercises.b_DataProcessingAndExploration.preprocessing import run_data_ingestion_and_processing
# from helpers.domino_short_id import domino_short_id


# # Configuration
# raw_filename = "raw_cc_transactions.csv"
# clean_filename = "preprocessing_processed_cc_transactions.csv"
# experiment_name = f"CC Fraud Preprocessing {domino_short_id()}"

# clean_df, clean_df_path, features_path, labels_path = run_data_ingestion_and_processing(
#     raw_filename=raw_filename,
#     clean_filename=clean_filename,
#     experiment_name=experiment_name
# )
# print(f"\n🎉 Processing complete!")
# print(f"   Features: {features_path}")
# print(f"   Labels: {labels_path}")
# print(f"   Clean data: {clean_df_path}")


# Path("/workflow/outputs/preprocessed_df_path").write_text(clean_df_path)

import os
import tempfile
from pathlib import Path
from domino import Domino
from exercises.b_DataProcessingAndExploration.preprocessing import run_data_ingestion_and_processing
from helpers.domino_short_id import domino_short_id


def get_domino_client():
    """Initialize Domino client with proper authentication."""
    return Domino(
        project=os.environ.get("DOMINO_PROJECT_NAME", "default-project"),
        api_key=os.environ.get("DOMINO_USER_API_KEY"),
        host=os.environ.get("DOMINO_API_HOST", "https://trial.dominodatalab.com")
    )


def upload_to_dataset(file_path, dataset_name, target_filename):
    """Upload a file to a Domino Dataset."""
    domino = get_domino_client()
    
    # Get the project's dataset ID
    datasets = domino.datasets_list()
    
    # Find or create the dataset
    dataset_id = None
    for dataset in datasets:
        if dataset['name'] == dataset_name:
            dataset_id = dataset['id']
            break
    
    if dataset_id is None:
        print(f"Creating new dataset: {dataset_name}")
        dataset_response = domino.datasets_create(dataset_name)
        dataset_id = dataset_response['id']
    
    # Upload the file to the dataset
    print(f"Uploading {file_path} to dataset {dataset_name}")
    domino.datasets_upload_files(
        dataset_id=dataset_id,
        file_path=file_path,
        target_relative_path=target_filename
    )
    
    # Get the dataset mount path for the workflow
    dataset_mount_path = f"/domino/datasets/{dataset_name}/{target_filename}"
    return dataset_mount_path


def main():
    # Configuration
    raw_filename = "raw_cc_transactions.csv"
    clean_filename = "preprocessing_processed_cc_transactions.csv"
    experiment_name = f"CC Fraud Preprocessing {domino_short_id()}"
    dataset_name = "processed-fraud-data"

    # Process the data
    clean_df, clean_df_path, features_path, labels_path = run_data_ingestion_and_processing(
        raw_filename=raw_filename,
        clean_filename=clean_filename,
        experiment_name=experiment_name
    )
    
    print(f"\n🎉 Processing complete!")
    print(f"   Features: {features_path}")
    print(f"   Labels: {labels_path}")
    print(f"   Clean data: {clean_df_path}")
    
    # Upload to Domino Dataset
    try:
        dataset_path = upload_to_dataset(
            file_path=clean_df_path,
            dataset_name=dataset_name,
            target_filename=clean_filename
        )
        print(f"✅ Uploaded to dataset: {dataset_path}")
        
        # Write the dataset path to workflow output
        Path("/workflow/outputs/preprocessed_df_path").write_text(dataset_path)
        
        return dataset_path
        
    except Exception as e:
        print(f"❌ Error uploading to dataset: {e}")
        # Fallback: try to use the local path
        print("🔄 Using local path as fallback...")
        Path("/workflow/outputs/preprocessed_df_path").write_text(clean_df_path)
        return clean_df_path


if __name__ == "__main__":
    main()