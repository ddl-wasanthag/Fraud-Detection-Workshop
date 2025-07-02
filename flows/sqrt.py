import os
from pathlib import Path

# Get working directory from environment or default
# workflow_dir = os.environ.get("WORKFLOW_DIR", "/workflow")
# inputs_dir = Path(os.environ.get("WORKFLOW_INPUTS_DIR", f"{workflow_dir}/inputs"))
# outputs_dir = Path(os.environ.get("WORKFLOW_OUTPUTS_DIR", f"{workflow_dir}/outputs"))

value = Path("/workflow/inputs/sum").read_text()
df = Path("/workflow/inputs/df").read_text()



# # Read input
# value = (inputs_dir / "value").read_text()
# input_df = (inputs_dir / "input_df").read_text()

print(f'sum {value}')
print(f'df {df}')


# Calculate square root
sqrt_result = int(value) ** 0.5
print(f"The square root of {value} is {sqrt_result}")

# Ensure output directory exists
outputs_dir.mkdir(parents=True, exist_ok=True)
# Write output
(outputs_dir / "sqrt").write_text(str(sqrt_result))