
# File: flows/b_training_ada.py
import os
import pandas as pd
from exercises.b_DataProcessingAndExploration.preprocessing import run_data_ingestion_and_processing
from helpers.domino_short_id import domino_short_id


def main():
    # Configuration
    raw_filename = "raw_cc_transactions.csv"
    clean_filename = "preprocessing_processed_cc_transactions.csv"
    experiment_name = f"CC Fraud Preprocessing {domino_short_id()}"

    # Process the data
    clean_df, _, _, _ = run_data_ingestion_and_processing(
        raw_filename=raw_filename,
        clean_filename=clean_filename,
        experiment_name=experiment_name
    )
    
    print(f"\n🎉 Processing complete!")
    print(f"   DataFrame shape: {clean_df.shape}")

    # Write the filename to workflow output
    with open("/workflow/outputs/preprocessed_df_path", "w") as f:
        f.write(clean_filename)
    
    print(f"✅ Filename saved: {clean_filename}")
    return clean_filename


if __name__ == "__main__":
    main()
