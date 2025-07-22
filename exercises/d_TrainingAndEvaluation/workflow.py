import os
from flytekit import workflow, task
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask
# DatasetSnapshot import left in case you actually use it elsewhere
from flytekitplugins.domino.task import DatasetSnapshot  


# --- lightweight python task to bundle the 3 string outputs into one JSON blob ---
@task
def consolidate_results(ada_results: str, gnb_results: str, xgb_results: str) -> str:
    """
    Take the raw stringified dicts coming back from each trainer and return a single JSON string.
    Safely handles single-quoted dicts via ast.literal_eval.
    """
    import ast, json
    def to_dict(s: str):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return ast.literal_eval(s)  # fallback if it's a python dict repr

    consolidated = {
        "AdaBoost": to_dict(ada_results),
        "GaussianNB": to_dict(gnb_results),
        "XGBoost": to_dict(xgb_results),
    }
    return json.dumps(consolidated)


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

    all_results_json = consolidate_results(
        ada_results=ada_results,
        gnb_results=gnb_results,
        xgb_results=xgb_results
    )

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
