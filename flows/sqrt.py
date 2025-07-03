import os
from pathlib import Path

outputs_dir = Path("/workflow/outputs")

# Read inputs matching the workflow task definition
value = Path("/workflow/inputs/value").read_text()  # Changed from "sum" to "value"
input_df = Path("/workflow/inputs/input_df").read_text()  # Changed from "df" to "input_df"

print(f'value: {value}')
print(f'input_df: {input_df}')

# Calculate square root
sqrt_result = int(value) ** 0.5
print(f"The square root of {value} is {sqrt_result}")

# Ensure output directory exists
outputs_dir.mkdir(parents=True, exist_ok=True)
# Write output (name matches task definition)
(outputs_dir / "sqrt").write_text(str(sqrt_result))