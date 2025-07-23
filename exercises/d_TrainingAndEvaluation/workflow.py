# workflow.py
import os
from flytekit import workflow, task
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask
from flytekitplugins.domino.task import DatasetSnapshot  


@workflow
def credit_card_fraud_detection_workflow() -> str:
    transformed_filename = 'transformed_cc_transactions.csv'

    ada_training_task = DominoJobTask(
        name='Train AdaBoost classifier',
        domino_job_config=DominoJobConfig(Command="python exercises/d_TrainingAndEvaluation/trainer_anb.py"),
        inputs={'transformed_filename': str},
        outputs={'results': str},
        use_latest=True,
        cache=True
    )

    gnb_training_task = DominoJobTask(
        name='Train GaussianNB classifier',
        domino_job_config=DominoJobConfig(Command="python exercises/d_TrainingAndEvaluation/trainer_gnb.py"),
        inputs={'transformed_filename': str},
        outputs={'results': str},
        use_latest=True,
        cache=True
    )

    ada_results = ada_training_task(transformed_filename=transformed_filename)
    gnb_results = gnb_training_task(transformed_filename=transformed_filename)

    compare_task = DominoJobTask(
        name='Compare training results',
        domino_job_config=DominoJobConfig(Command="python exercises/d_TrainingAndEvaluation/compare_training_results.py"),
        inputs={'ada_results': str, 'gnb_results': str},
        outputs={'comparison': str},
        use_latest=True
    )

    comparison = compare_task(ada_results=ada_results, gnb_results=gnb_results)

    print('flow comparison:')
    print(comparison)

    # Return whatever you want Flyte to show as the final output. CSV is convenient.
    return comparison


