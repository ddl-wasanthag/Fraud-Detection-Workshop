
# File: flows/b_training_ada.py
import os
import pandas as pd
from domino_data.data_sources import DataSourceClient
from exercises.b_DataProcessingAndExploration.preprocessing import run_data_ingestion_and_processing
from helpers.domino_short_id import domino_short_id


def save_to_data_source(df, filename):
    """Save DataFrame to Domino Data Source."""
    try:
        # Try to use the existing data source client
        ds_client = DataSourceClient()
        
        # Get the data source (assuming it's already configured)
        ds = ds_client.get_datasource("credit_card_fraud_detection")
        
        # Convert DataFrame to CSV bytes
        csv_data = df.to_csv(index=False).encode('utf-8')
        
        # Upload to data source
        print(f"📤 Uploading {filename} to data source")
        ds.upload_fileobj(filename, csv_data)
        
        print(f"✅ Successfully uploaded to data source: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Error with data source: {e}")
        raise


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
    
    # Save to Data Source
    uploaded_filename = save_to_data_source(clean_df, clean_filename)
    
    # Write the filename to workflow output
    with open("/workflow/outputs/preprocessed_df_path", "w") as f:
        f.write(uploaded_filename)
    
    print(f"✅ Filename saved: {uploaded_filename}")
    return uploaded_filename


if __name__ == "__main__":
    main()
