from pathlib import Path
import pandas as pd

preprocessed_df_path = Path("/workflow/inputs/preprocessed_df_path").read_text()

print(preprocessed_df_path)

test_df = pd.DataFrame({
    "transaction_id": [1, 2, 3],
    "amount": [100.0, 200.0, 300.0],
    "is_fraud": [0, 0, 1]
})

# Write output
Path("/workflow/outputs/results_df").write_text(str(test_df))
