import os
from pathlib import Path

# Get working directory from environment or default
workflow_dir = os.environ.get("WORKFLOW_DIR", "/workflow")
inputs_dir = Path(os.environ.get("WORKFLOW_INPUTS_DIR", f"{workflow_dir}/inputs"))
outputs_dir = Path(os.environ.get("WORKFLOW_OUTPUTS_DIR", f"{workflow_dir}/outputs"))

# Read inputs
first_value = (inputs_dir / "first_value").read_text()
second_value = (inputs_dir / "second_value").read_text()

# Calculate sum
sum_result = int(first_value) + int(second_value)
print(f"The sum of {first_value} + {second_value} is {sum_result}")

# Ensure output directory exists
outputs_dir.mkdir(parents=True, exist_ok=True)
# Write output
(outputs_dir / "sum").write_text(str(sum_result))