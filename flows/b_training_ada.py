from pathlib import Path


preprocessed_df = Path("/workflow/inputs/preprocessed_df").read_text()

print(preprocessed_df)

# Write output
Path("/workflow/outputs/results_df").write_text(str(preprocessed_df))
