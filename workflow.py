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
    gnb_training_task = DominoJobTask(
        name='Train classifier (GaussianNB)',
        domino_job_config=DominoJobConfig(Command="python flows/b_training_gnb.py"),
        inputs={'preprocessed_df_path': str},
        outputs={'results_df': str},
        use_latest=True
    )
    xgb_training_task = DominoJobTask(
        name='Train classifier (XGBoost)',
        domino_job_config=DominoJobConfig(Command="python flows/b_training_xgb.py"),
        inputs={'preprocessed_df_path': str},
        outputs={'results_df': str},
        use_latest=True
    )

    training_results_ada = ada_training_task(preprocessed_df_path=preprocessed_df_path)
    training_results_gnb = gnb_training_task(preprocessed_df_path=preprocessed_df_path)
    training_results_xgb = xgb_training_task(preprocessed_df_path=preprocessed_df_path)

    return training_results_ada

