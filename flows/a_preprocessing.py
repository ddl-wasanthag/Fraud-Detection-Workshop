from flows.a_preprocessing_old import run_data_ingestion_and_processing
from exercises.b_DataProcessingAndExploration.preprocessing import run_data_ingestion_and_processing
from helpers.domino_short_id import domino_short_id


# Configuration
raw_filename = "raw_cc_transactions.csv"  # Fixed: removed trailing comma
clean_filename = "preprocessing_processed_cc_transactions.csv"
experiment_name = f"CC Fraud Preprocessing {domino_short_id()}"

# Run the pipeline
try:
    clean_df, clean_path, features_path, labels_path = run_data_ingestion_and_processing(
        raw_filename=raw_filename,
        clean_filename=clean_filename,
        experiment_name=experiment_name
    )
    print(f"\n🎉 Processing complete!")
    print(f"   Features: {features_path}")
    print(f"   Labels: {labels_path}")
    print(f"   Clean data: {clean_path}")
    print(f"   Clean data (head): {clean_df.head()}")

    Path("/workflow/outputs/preprocessed_df").write_text(clean_df.to_json(orient='records'))

except Exception as e:
    print(f"❌ Pipeline failed: {str(e)}")
    raise

from pathlib import Path
a = 2
b = 3
# Calculate sum
sum = int(a) + int(b)
print(f"The sum of {a} + {b} is {sum}")

# Write output
Path("/workflow/outputs/sum").write_text(str(sum))