from flytekit import workflow
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask

@workflow
def simple_math_workflow(a: int, b: int) -> float:

    # Create first task
    add_task = DominoJobTask(
        name='Add numbers',
        domino_job_config=DominoJobConfig(Command="python add.py"),
        inputs={'first_value': int, 'second_value': int},
        outputs={'sum': int},
        use_latest=True
    )
    sum = add_task(first_value=a, second_value=b)

    # Create second task
    sqrt_task = DominoJobTask(
        name='Square root',
        domino_job_config=DominoJobConfig(Command="python sqrt.py"),
        inputs={'value': int},
        outputs={'sqrt': float},
        use_latest=True
    )
    sqrt = sqrt_task(value=sum)

    return sqrt

@workflow
def fraud_detection_best_practice_workflow() -> tuple:
    # Task 1: Data preprocessing
    preprocess_task = DominoJobTask(
        name='Preprocessing',
        domino_job_config=DominoJobConfig(Command="python flows/a_preprocessing.py"),
        inputs={},
        outputs={'processed_data_path': str},  # Explicit output: path to processed data
        use_latest=True
    )
    preprocess_result = preprocess_task()
    processed_data_path = preprocess_result['processed_data_path']

    # Task 2: Model training (parallel, all depend on processed_data_path)
    ada_task = DominoJobTask(
        name='Train AdaBoost',
        domino_job_config=DominoJobConfig(Command="python flows/b_training_ada.py"),
        inputs={'processed_data_path': str},
        outputs={'model_path': str},
        use_latest=True
    )
    gnb_task = DominoJobTask(
        name='Train GaussianNB',
        domino_job_config=DominoJobConfig(Command="python flows/b_training_gnb.py"),
        inputs={'processed_data_path': str},
        outputs={'model_path': str},
        use_latest=True
    )
    xgb_task = DominoJobTask(
        name='Train XGBoost',
        domino_job_config=DominoJobConfig(Command="python flows/b_training_xgb.py"),
        inputs={'processed_data_path': str},
        outputs={'model_path': str},
        use_latest=True
    )

    ada_result = ada_task(processed_data_path=processed_data_path)
    gnb_result = gnb_task(processed_data_path=processed_data_path)
    xgb_result = xgb_task(processed_data_path=processed_data_path)

    return ada_result['model_path'], gnb_result['model_path'], xgb_result['model_path']
