
# File: flows/b_training_ada.py
import os
from exercises.b_DataProcessingAndExploration.preprocessing import run_data_ingestion_and_processing
from helpers.domino_short_id import domino_short_id


# Configuration
raw_filename = "raw_cc_transactions.csv" 
clean_filename = "preprocessing_processed_cc_transactions.csv"
experiment_name = f"CC Fraud Preprocessing {domino_short_id()}"

# Run the pipeline
try:
    clean_df, clean_path, features_path, labels_path = run_data_ingestion_and_processing(
        raw_filename=raw_filename,
        clean_filename=clean_filename,
        experiment_name=experiment_name
    )
    print(f"   Features: {features_path}")
    print(f"   Labels: {labels_path}")
    print(f"   Clean data: {clean_path}")
    print(f"   Clean data (head): {clean_df.head()}")
    print(f"   DataFrame shape: {clean_df.shape}")

    # Write the filename to workflow output
    with open("/workflow/outputs/preprocessed_df_path", "w") as f:
        f.write(clean_filename)

except Exception as e:
    print(f"❌ Pipeline failed: {str(e)}")
    raise
