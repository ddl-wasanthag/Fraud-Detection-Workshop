from pathlib import Path
from exercises.b_DataProcessingAndExploration.preprocessing import run_data_ingestion_and_processing
from helpers.domino_short_id import domino_short_id


# Configuration
raw_filename = "raw_cc_transactions.csv"
clean_filename = "preprocessing_processed_cc_transactions.csv"
experiment_name = f"CC Fraud Preprocessing {domino_short_id()}"

# clean_df, clean_path, features_path, labels_path = run_data_ingestion_and_processing(
#     raw_filename=raw_filename,
#     clean_filename=clean_filename,
#     experiment_name=experiment_name
# )
# print(f"\n🎉 Processing complete!")
# print(f"   Features: {features_path}")
# print(f"   Labels: {labels_path}")
# print(f"   Clean data: {clean_path}")
# print(f"   Clean data (head): {clean_df.head()}")

import pandas as pd
clean_df2 = pd.DataFrame({
    "transaction_id": [1, 2, 3],
    "amount": [100.0, 200.0, 300.0],
    "is_fraud": [0, 0, 1]
}) 

print(clean_df2.head())
# print(clean_df.head())

print(len(clean_df2))
# print(len(clean_df))

Path("/workflow/outputs/preprocessed_df").write_text(clean_df2.head(10).to_json(orient='records'))
