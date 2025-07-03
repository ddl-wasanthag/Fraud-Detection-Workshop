from pathlib import Path

# Read input
preprocessed_df = Path("/workflow/inputs/preprocessed_df").read_text()
print('my preprocessed_df here should be a string')
print(preprocessed_df)
# Calculate square root
sqrt = int(32) ** 0.5
print(f"The square root of {32} is {sqrt}")

# Write output
Path("/workflow/outputs/sqrt").write_text(str(sqrt))
