# workflow.py
from flytekit import workflow
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask
from typing import NamedTuple



@workflow
def credit_card_fraud_detection_workflow() -> str:

    preprocessing_task = DominoJobTask(
        name='Preprocess and scale inputs',
        domino_job_config=DominoJobConfig(Command="python flows/a_preprocessing.py"),
        inputs={},
        outputs={'preprocessed_df_path': str,},
        use_latest=True,
        cache=False
    )
    preprocessed_df_path = preprocessing_task()

    ada_training_task = DominoJobTask(
        name='Train classifier (AdaBoost)',
        domino_job_config=DominoJobConfig(Command="python flows/b_training_ada.py"),
        inputs={'preprocessed_df_path': str},
        outputs={'results_df': str},
        use_latest=True
    )
    # gnb_training_task = DominoJobTask(
    #     name='Train classifier (GaussianNB)',
    #     domino_job_config=DominoJobConfig(Command="python flows/b_training_gnb.py"),
    #     inputs={'preprocessed_df': str},
    #     outputs={'results_df': str},
    #     use_latest=True
    # )

    # xgb_training_task = DominoJobTask(
    #     name='Train classifier (XGBoost)',
    #     domino_job_config=DominoJobConfig(Command="python flows/b_training_xgb.py"),
    #     inputs={'preprocessed_df': str},
    #     outputs={'results_df': str},
    #     use_latest=True
    # )

    training_results_ada = ada_training_task(preprocessed_df_path=preprocessed_df_path)
    # training_results_gnb = gnb_training_task(preprocessed_df=preprocessed_df)
    # training_results_xgb = xgb_training_task(preprocessed_df=preprocessed_df)

    return training_results_ada


# @workflow
# def credit_card_fraud_detection_workflow2() -> float:

#     # Create first task
#     preprocessing_task = DominoJobTask(
#         name="preprocessing_task",
#         domino_job_config=DominoJobConfig(Command="python flows/a_preprocessing.py"),
#         inputs={},
#         outputs={"preprocessed_df": str},
#         use_latest=True,
#     )
#     preprocessed_df = preprocessing_task()

#     # sum_val, df = result["sum"], result["df"]

#     # Create second task
#     sqrt_task = DominoJobTask(
#         name="sqrt_task",
#         domino_job_config=DominoJobConfig(Command="python flows/sqrt.py"),
#         inputs={"value": int, "input_df": str},
#         outputs={"sqrt": float},
#         use_latest=True,
#     )
#     sqrt_result = sqrt_task(value=4, input_df=c)

#     return sqrt_result

# class FraudDetectionResults(NamedTuple):
#     ada_model_path: str
#     gnb_model_path: str
#     xgb_model_path: str

# @workflow
# def fraud_detection_best_practice_workflow() -> FraudDetectionResults:
#     # Task 1: Data preprocessing
#     preprocess_task = DominoJobTask(
#         name='Preprocessing',
#         domino_job_config=DominoJobConfig(Command="python flows/a_preprocessing.py"),
#         inputs={},
#         outputs={'processed_data_path': str},
#         use_latest=True
#     )
#     processed_data_path = preprocess_task()

#     # Task 2: Model training (parallel, all depend on processed_data_path)
#     ada_task = DominoJobTask(
#         name='Train AdaBoost',
#         domino_job_config=DominoJobConfig(Command="python flows/b_training_ada.py"),
#         inputs={'processed_data_path': str},
#         outputs={'model_path': str},
#         use_latest=True
#     )
#     gnb_task = DominoJobTask(
#         name='Train GaussianNB',
#         domino_job_config=DominoJobConfig(Command="python flows/b_training_gnb.py"),
#         inputs={'processed_data_path': str},
#         outputs={'model_path': str},
#         use_latest=True
#     )
#     xgb_task = DominoJobTask(
#         name='Train XGBoost',
#         domino_job_config=DominoJobConfig(Command="python flows/b_training_xgb.py"),
#         inputs={'processed_data_path': str},
#         outputs={'model_path': str},
#         use_latest=True
#     )

#     ada_model_path = ada_task(processed_data_path=processed_data_path)
#     gnb_model_path = gnb_task(processed_data_path=processed_data_path)
#     xgb_model_path = xgb_task(processed_data_path=processed_data_path)

#     return FraudDetectionResults(
#         ada_model_path=ada_model_path,
#         gnb_model_path=gnb_model_path,
#         xgb_model_path=xgb_model_path
#     )
