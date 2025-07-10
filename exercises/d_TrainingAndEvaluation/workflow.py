from flytekit import workflow, task
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask


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

# Fixed credit card workflow
@task
def preprocessing_task() -> str:
    return DominoJobTask(
        name='Feature engineering and preprocessing',
        domino_job_config=DominoJobConfig(Command="python exercises/c_DataEngineering/data_engineering.py"),
        inputs={},
        outputs={'transformed_filepath': str},
        use_latest=True,
        cache=False
    )()

@task
def ada_training_task(transformed_filepath: str) -> str:
    return DominoJobTask(
        name='Train AdaBoost classifier',
        domino_job_config=DominoJobConfig(Command="python exercises/trainer_ada.py"),
        inputs={'transformed_filepath': str},
        outputs={'results': str},
        use_latest=True
    )(transformed_filepath=transformed_filepath)

@task
def gnb_training_task(transformed_filepath: str) -> str:
    return DominoJobTask(
        name='Train GaussianNB classifier',
        domino_job_config=DominoJobConfig(Command="python exercises/trainer_gnb.py"),
        inputs={'transformed_filepath': str},
        outputs={'results': str},
        use_latest=True
    )(transformed_filepath=transformed_filepath)

@task
def xgb_training_task(transformed_filepath: str) -> str:
    return DominoJobTask(
        name='Train XGBoost classifier',
        domino_job_config=DominoJobConfig(Command="python exercises/trainer_xgb.py"),
        inputs={'transformed_filepath': str},
        outputs={'results': str},
        use_latest=True
    )(transformed_filepath=transformed_filepath)

@workflow
def credit_card_fraud_detection_workflow() -> str:
    preprocessing_task = DominoJobTask(
        name='Feature engineering and preprocessing',
        domino_job_config=DominoJobConfig(Command="python exercises/c_DataEngineering/data_engineering.py"),
        inputs={},
        outputs={'transformed_filepath': str},
        use_latest=True,
        cache=False
    )
    
    transformed_filepath = preprocessing_task()

    ada_training_task = DominoJobTask(
            name='Train AdaBoost classifier',
            domino_job_config=DominoJobConfig(Command="python exercises/trainer_ada.py"),
            inputs={'transformed_filepath': str},
            outputs={'results': str},
            use_latest=True
        )
    gnb_training_task = DominoJobTask(
            name='Train GaussianNB classifier',
            domino_job_config=DominoJobConfig(Command="python exercises/trainer_gnb.py"),
            inputs={'transformed_filepath': str},
            outputs={'results': str},
            use_latest=True
        )
    xgb_training_task = DominoJobTask(
            name='Train XGBoost classifier',
            domino_job_config=DominoJobConfig(Command="python exercises/trainer_xgb.py"),
            inputs={'transformed_filepath': str},
            outputs={'results': str},
            use_latest=True
        )

    ada_results = ada_training_task(transformed_filepath=transformed_filepath)
    gnb_results = gnb_training_task(transformed_filepath=transformed_filepath)
    xgb_results = xgb_training_task(transformed_filepath=transformed_filepath)
    
    return ada_results