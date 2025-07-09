from flytekit import workflow
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask

@workflow
def simple_math_workflow() -> float:

    # Create first task
    add_task = DominoJobTask(
        name='Add numbers',
        domino_job_config=DominoJobConfig(Command="python add.py"),
        inputs={},
        outputs={'sum': int},
        use_latest=True
    )
    sum = add_task()

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

# # Fixed credit card workflow
# @task
# def preprocessing_task() -> str:
#     return DominoJobTask(
#         name='Feature engineering and preprocessing',
#         domino_job_config=DominoJobConfig(Command="python exercises/DataEngineering/feature_engineering.py"),
#         inputs={},
#         outputs={'transformed_filename': str},
#         use_latest=True,
#         cache=False
#     )()

# @task
# def ada_training_task(transformed_filename: str) -> str:
#     return DominoJobTask(
#         name='Train AdaBoost classifier',
#         domino_job_config=DominoJobConfig(Command="python exercises/trainer_ada.py"),
#         inputs={'transformed_filename': str},
#         outputs={'results': str},
#         use_latest=True
#     )(transformed_filename=transformed_filename)

# @task
# def gnb_training_task(transformed_filename: str) -> str:
#     return DominoJobTask(
#         name='Train GaussianNB classifier',
#         domino_job_config=DominoJobConfig(Command="python exercises/trainer_gnb.py"),
#         inputs={'transformed_filename': str},
#         outputs={'results': str},
#         use_latest=True
#     )(transformed_filename=transformed_filename)

# @task
# def xgb_training_task(transformed_filename: str) -> str:
#     return DominoJobTask(
#         name='Train XGBoost classifier',
#         domino_job_config=DominoJobConfig(Command="python exercises/trainer_xgb.py"),
#         inputs={'transformed_filename': str},
#         outputs={'results': str},
#         use_latest=True
#     )(transformed_filename=transformed_filename)

# @workflow
# def credit_card_fraud_detection_workflow() -> str:
#     transformed_filename = preprocessing_task()
    
#     ada_results = ada_training_task(transformed_filename=transformed_filename)
#     gnb_results = gnb_training_task(transformed_filename=transformed_filename)
#     xgb_results = xgb_training_task(transformed_filename=transformed_filename)
    
#     return ada_results