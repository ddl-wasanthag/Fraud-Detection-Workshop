from pathlib import Path

a = 33
b = 21

# Calculate sum
sum = int(a) + int(b)
print(f"The sum of {a} + {b} is {sum}")

# Write output
Path("/workflow/outputs/sum").write_text(str(sum))