from flytekit import workflow
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask
import os

# Centralize workflow directory logic
WORKFLOW_DIR = os.environ.get("WORKFLOW_DIR", "/workflow")
INPUTS_DIR = os.environ.get("WORKFLOW_INPUTS_DIR", f"{WORKFLOW_DIR}/inputs")
OUTPUTS_DIR = os.environ.get("WORKFLOW_OUTPUTS_DIR", f"{WORKFLOW_DIR}/outputs")

@workflow
def simple_math_workflow(a: int, b: int) -> float:
    # Create first task
    add_task = DominoJobTask(
        name='Add numbers',
        domino_job_config=DominoJobConfig(Command="python flows/add.py"),
        inputs={'first_value': int, 'second_value': int},
        outputs={'sum': int},
        use_latest=True
    )
    sum_result = add_task(first_value=a, second_value=b)

    # Create second task
    sqrt_task = DominoJobTask(
        name='Square root',
        domino_job_config=DominoJobConfig(Command="python flows/sqrt.py"),
        inputs={'value': int},
        outputs={'sqrt': float},
        use_latest=True
    )
    sqrt_result = sqrt_task(value=sum_result)

    return sqrt_result