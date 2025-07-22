# workflow.py
import os
from flytekit import workflow, task
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask
from flytekitplugins.domino.task import DatasetSnapshot  


@workflow
def credit_card_fraud_detection_workflow() -> str:
    transformed_filename = 'transformed_cc_transactions.csv'

    gnb_training_task = DominoJobTask(
        name='Train GaussianNB classifier',
        domino_job_config=DominoJobConfig(Command="python exercises/d_TrainingAndEvaluation/trainer_gnb.py"),
        inputs={'transformed_filename': str},
        outputs={'results': str},
        use_latest=True
    )

    gnb_results = gnb_training_task(transformed_filename=transformed_filename)

    compare_task = DominoJobTask(
        name='Compare training results',
        domino_job_config=DominoJobConfig(Command="python exercises/d_TrainingAndEvaluation/compare_training_results.py"),
        inputs={'results_json': str},
        outputs={'comparison': str},
        use_latest=True
    )

    comparison_csv = compare_task(results_json=all_results_json)

    # Return whatever you want Flyte to show as the final output. CSV is convenient.
    return comparison_csv
