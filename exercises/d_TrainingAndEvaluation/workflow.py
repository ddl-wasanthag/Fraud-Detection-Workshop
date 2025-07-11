import os
from domino import Domino
from flytekit import workflow, task
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask, DatasetSnapshot


@workflow
def simple_math_workflow() -> float:

    # Create first task
    add_task = DominoJobTask(
        name='Add numbers',
        domino_job_config=DominoJobConfig(Command="python flows_test_to_remove/add.py"),
        inputs={},
        outputs={'sum': int},
        use_latest=True
    )
    sum = add_task()

    # Create second task
    sqrt_task = DominoJobTask(
        name='Square root',
        domino_job_config=DominoJobConfig(Command="python flows_test_to_remove/sqrt.py"),
        inputs={'value': int},
        outputs={'sqrt': float},
        use_latest=True
    )
    sqrt = sqrt_task(value=sum)

    return sqrt


@workflow
def credit_card_fraud_detection_workflow() -> str:
    transformed_filename = 'transformed_cc_transactions.csv'
   
    ada_training_task = DominoJobTask(
        name='Train AdaBoost classifier',
        domino_job_config=DominoJobConfig(Command="python exercises/d_TrainingAndEvaluation/trainer_ada.py"),
        inputs={'transformed_filename': str},
        outputs={'results': str},
        use_latest=True
    )
    gnb_training_task = DominoJobTask(
            name='Train GaussianNB classifier',
            domino_job_config=DominoJobConfig(Command="python exercises/d_TrainingAndEvaluation/trainer_gnb.py"),
            inputs={'transformed_filename': str},
            outputs={'results': str},
            use_latest=True
        )
    xgb_training_task = DominoJobTask(
            name='Train XGBoost classifier',
            domino_job_config=DominoJobConfig(Command="python exercises/d_TrainingAndEvaluation/trainer_xgb.py"),
            inputs={'transformed_filename': str},
            outputs={'results': str},
            use_latest=True
        )

    ada_results = ada_training_task(transformed_filename=transformed_filename)
    gnb_results = gnb_training_task(transformed_filename=transformed_filename)
    xgb_results = xgb_training_task(transformed_filename=transformed_filename)

    return ada_results