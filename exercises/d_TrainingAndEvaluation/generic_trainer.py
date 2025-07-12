import os, time, yaml
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, precision_recall_curve,
    confusion_matrix
)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from domino_short_id import domino_short_id
from flytekitplugins.domino.artifact import Artifact, DATA, MODEL, REPORT


# Directories
experiment_name = f"CC Fraud Classifier Training {domino_short_id()}"
domino_working_dir = os.environ.get("DOMINO_WORKING_DIR", ".")
domino_project_name = os.environ.get("DOMINO_PROJECT_NAME", "my-local-project")
domino_artifact_dir = domino_working_dir.replace('code', 'artifacts')
domino_dataset_dir = f"{domino_working_dir.replace('code', 'data')}/{domino_project_name}"

ModelArtifact = Artifact(name="Fraud Detection Models", type=MODEL)
DataArtifact = Artifact(name="Training Data", type=DATA)
ReportArtifact = Artifact(name="Model Reports", type=REPORT)



def save_domino_artifacts(name: str, metrics: dict, model_path: str = None):
    # Create artifacts directory for Domino
    domino_artifacts_path = Path("/workflow/outputs")
    domino_artifacts_path.mkdir(exist_ok=True, parents=True)
    
    # Save model summary report
    report_path = domino_artifacts_path / f"{name.lower().replace(' ', '_')}_report.json"
    with open(report_path, 'w') as f:
        import json
        json.dump(metrics, f, indent=2)
    
    # Save model artifact reference if model path provided
    if model_path and Path(model_path).exists():
        model_artifact_path = domino_artifacts_path / f"{name.lower().replace(' ', '_')}_model.pkl"
        import shutil
        shutil.copy2(model_path, model_artifact_path)

def train_and_log(
    model, name: str,
    df: pd.DataFrame,
    X_train: pd.DataFrame, X_val: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series,
    features: list
):
    """
    Train the model, log parameters, metrics, plots, and model artifact to MLflow.
    """
    # Ensure artifact directory
    Path(domino_artifact_dir).mkdir(exist_ok=True, parents=True)

    with mlflow.start_run(run_name=name):
        # Log model hyperparameters
        mlflow.log_param("model_name", model.__class__.__name__)
        mlflow.log_param("num_features", len(features))
        mlflow.log_param("num_rows", len(df))

        # Save params YAML to artifacts
        params_yaml = {
            "model_name": model.__class__.__name__,
            "num_features": len(features),
            "num_rows": len(df),
            "features": features,
        }
        yaml_path = os.path.join(domino_artifact_dir, f"{name.lower().replace(' ', '_')}_params.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(params_yaml, f, default_flow_style=False)
        mlflow.log_artifact(yaml_path, artifact_path="params")

        # Fit
        start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start

        # Predictions & metrics
        proba = model.predict_proba(X_val)[:, 1]
        pred = model.predict(X_val)
        metrics = {
            "roc_auc": roc_auc_score(y_val, proba),
            "pr_auc": average_precision_score(y_val, proba),
            "accuracy": accuracy_score(y_val, pred),
            "precision_fraud": precision_score(y_val, pred, pos_label=1),
            "recall_fraud": recall_score(y_val, pred, pos_label=1),
            "f1_fraud": f1_score(y_val, pred, pos_label=1),
            "fit_time_sec": fit_time,
        }
        mlflow.log_metrics(metrics)

        # Save small summary files to artifacts (not full predictions)
        summary_metrics = {
            'model_name': name,
            'validation_samples': len(y_val),
            'fraud_samples': sum(y_val),
            **metrics
        }
        
        # Save metrics summary as CSV to artifacts
        metrics_df = pd.DataFrame([summary_metrics])
        metrics_csv_path = os.path.join(domino_artifact_dir, f"{name.lower().replace(' ', '_')}_metrics.csv")
        print('metrics metrics_csv_path', metrics_csv_path)
        metrics_df.to_csv(metrics_csv_path, index=False)
        mlflow.log_artifact(metrics_csv_path, artifact_path="metrics")

        model_pkl_path = os.path.join(domino_artifact_dir, f"{name.lower().replace(' ', '_')}_model.pkl")
        import joblib
        joblib.dump(model, model_pkl_path)

        # Inference signature & model logging
        signature = infer_signature(X_val, proba)
        input_example = X_val.iloc[:5]
        mlflow.sklearn.log_model(
            model,
            artifact_path=f"{name.lower().replace(' ', '_')}_model",
            signature=signature,
            input_example=input_example
        )
        mlflow.set_tag("pipeline", "classifier_training")
        mlflow.set_tag("model", name)

        # Plotting helpers
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_val, proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{name} (AUC={metrics['roc_auc']:.3f})")
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        roc_path = os.path.join(domino_artifact_dir, f"{name.lower().replace(' ', '_')}_roc.png")
        plt.tight_layout(); plt.savefig(roc_path); plt.close()
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # Precision-Recall Curve
        rec, prec, _ = precision_recall_curve(y_val, proba)
        plt.figure()
        plt.plot(rec, prec, label=name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        pr_path = os.path.join(domino_artifact_dir, f"{name.lower().replace(' ', '_')}_pr.png")
        plt.tight_layout(); plt.savefig(pr_path); plt.close()
        mlflow.log_artifact(pr_path, artifact_path="plots")

        # Confusion Matrix
        cm = confusion_matrix(y_val, pred, normalize='true')
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=[0,1], yticklabels=[0,1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Normalized Confusion Matrix')
        cm_path = os.path.join(domino_artifact_dir, f"{name.lower().replace(' ', '_')}_cm.png")
        plt.tight_layout(); plt.savefig(cm_path); plt.close()
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Feature importances (if available)
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            idx = np.argsort(imp)[::-1][:15]
            plt.figure()
            plt.bar(range(len(idx)), imp[idx])
            plt.xticks(range(len(idx)), [features[i] for i in idx], rotation=45, ha='right')
            plt.title('Top Feature Importances')
            fi_path = os.path.join(domino_artifact_dir, f"{name.lower().replace(' ', '_')}_fi.png")
            plt.tight_layout(); plt.savefig(fi_path); plt.close()
            mlflow.log_artifact(fi_path, artifact_path="plots")

        save_domino_artifacts(name, summary_metrics, model_pkl_path)

    mlflow.end_run()
    
    return {
        "model_name": name,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "accuracy": metrics["accuracy"],
        "precision_fraud": metrics["precision_fraud"],
        "recall_fraud": metrics["recall_fraud"],
        "f1_fraud": metrics["f1_fraud"],
        "fit_time_sec": metrics["fit_time_sec"],
    }

def train_fraud(model_obj, model_name, transformed_df_filename, random_state=None):

    # Set up experiment
    mlflow.set_experiment(experiment_name)

    # Load transformed data
    transformed_df_path = f"{domino_dataset_dir}/{transformed_df_filename}"
    transformed_df = pd.read_csv(transformed_df_path)
    
    # Split data 
    labels = "Class"
    df = transformed_df.dropna(subset=[labels]).copy()
    features = [c for c in df.columns if c != labels]
    X = df[features]
    y = df[labels]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    data_summary = {
        'total_samples': len(df),
        'fraud_samples': sum(y),
        'fraud_rate': sum(y) / len(y),
        'features': features,
        'train_samples': len(X_train),
        'val_samples': len(X_val)
    }
    
    domino_artifacts_path = Path("/workflow/outputs")
    domino_artifacts_path.mkdir(exist_ok=True, parents=True)
    
    data_summary_path = domino_artifacts_path / "data_summary.json"
    with open(data_summary_path, 'w') as f:
        import json
        json.dump(data_summary, f, indent=2)

    # Train model
    print(f'training model {model_name}')
    res = train_and_log(
        model_obj, model_name,
        df, X_train, X_val, y_train, y_val,
        features
    )
    return res

