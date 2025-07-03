from pathlib import Path

# Read input
value = Path("/workflow/inputs/value").read_text()
print('my value here should be a string')
print(value)
# Calculate square root
sqrt = int(22) ** 0.5
print(f"The square root of {value} is {sqrt}")

# Write output
Path("/workflow/outputs/sqrt").write_text(str(sqrt))
