from pathlib import Path
print('here here in sqrt')
# Read input
preprocessed_df = Path("/workflow/inputs/preprocessed_df").read_text()
print('my value here should be a string')
print(preprocessed_df)
# Calculate square root
sqrt = int(232) ** 0.5
print(f"The square root of {232} is {sqrt}")

# Write output
Path("/workflow/outputs/sqrt").write_text(str(sqrt))
