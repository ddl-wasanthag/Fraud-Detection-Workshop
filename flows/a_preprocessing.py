# Data Ingestion, Processing, and MLflow Model Logging
import io, os, time, yaml
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import mlflow
from mlflow.models import infer_signature

from domino_data.data_sources import DataSourceClient
from helpers.domino_short_id import domino_short_id
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ydata_profiling import ProfileReport

domino_working_dir = os.environ.get("DOMINO_WORKING_DIR", ".")
domino_datasource_dir = domino_working_dir.replace('code', 'data')
domino_artifact_dir = domino_working_dir.replace('code', 'artifacts')
domino_project_name = os.environ.get("DOMINO_PROJECT_NAME", "my-local-project")


def run_data_ingestion_and_processing(raw_filename, clean_filename, experiment_name):
    mlflow.set_experiment(experiment_name)

    # Ensure raw_filename is a string, not a list or array
    if isinstance(raw_filename, (list, tuple)):
        if len(raw_filename) == 0:
            raise ValueError("raw_filename must not be empty")
        raw_filename = raw_filename[0]

    ds = DataSourceClient().get_datasource("credit_card_fraud_detection")
    buf = io.BytesIO()
    ds.download_fileobj(raw_filename, buf)
    buf.seek(0)
    df = pd.read_csv(buf)
    print(f"🔍 Loaded {len(df):,} rows from {raw_filename}")

    # 2) Drop missing rows
    before = len(df)
    df = df.dropna()
    after = len(df)
    pct_removed = 100 * (before - after) / before if before > 0 else 0
    print(f"🧹 Dropped {before - after:,} rows with missing data")

    # 3) Match run_all: drop Class, Time, Hour from X
    X = df.drop(columns=["Class", "Time", "Hour"], errors="ignore")
    y = df["Class"]

    # 4) Detect numeric and categorical columns as in run_all
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=[object, "category"]).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features,)
    ])
    pipeline = Pipeline([
        ("preproc", preprocessor)
    ])
    start_time = time.time()
    features_processed = pipeline.fit_transform(X)
    fit_time = time.time() - start_time

    # 5) Save processed data exactly as in run_all
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    print("[DEBUG] Directory structure:")
    for root, dirs, files in os.walk(os.getcwd()):
        level = root.replace(os.getcwd(), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")
    # Ensure output directory exists before saving files
    output_dir = f"{domino_datasource_dir}/{domino_project_name}"
    os.makedirs(output_dir, exist_ok=True)
    features_path = f"{output_dir}/preprocessing_features_processed.npy"
    labels_path = f"{output_dir}/preprocessing_feature_labels.csv"
    np.save(features_path, features_processed)
    y.to_csv(labels_path, index=False)
    print(f"✅ Saved {features_path} and {labels_path} for downstream modeling")

    if hasattr(features_processed, "toarray"):
        X_arr = features_processed.toarray()
    else:
        X_arr = features_processed
    num_cols = numeric_features
    cat_cols = pipeline.named_steps["preproc"] \
                       .named_transformers_["cat"] \
                       .get_feature_names_out(categorical_features).tolist()
    all_cols = num_cols + cat_cols

    df_scaled = pd.DataFrame(X_arr, columns=all_cols)
    df_scaled["Class"] = y.values  # add back target if you like

    # 2) Save it under clean_filename
    clean_path = f"{domino_datasource_dir}/{domino_project_name}/{clean_filename}"
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df_scaled.to_csv(clean_path, index=False)
    print(f"[DEBUG] Saved clean file: {clean_path}")

    # Write output for Domino Flow contract if running as a flow
    if os.environ.get("DOMINO_IS_WORKFLOW_JOB", "false").lower() == "true":
        flow_output_path = "/workflow/outputs/processed_data_path"
        import shutil
        shutil.copyfile(clean_path, flow_output_path)
        print(f"[DEBUG] Wrote output for flow: {flow_output_path}")
        # Double-check existence
        print(f"[DEBUG] Output file exists: {os.path.exists(flow_output_path)} at {flow_output_path}")
        if not os.path.exists(flow_output_path):
            raise FileNotFoundError(f"Expected flow output file not found: {flow_output_path}")

    # Check all output files exist before returning
    for path in [features_path, labels_path, clean_path]:
        print(f"[DEBUG] Output file exists: {os.path.exists(path)} at {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected output file not found: {path}")

    if mlflow.active_run():
        mlflow.end_run()

    # 6) Start MLflow run and log everything
    with mlflow.start_run(run_name="Preprocessing Pipeline") as run:
        mlflow.log_artifact(clean_path, artifact_path="data")
        mlflow.log_param("raw_filename", raw_filename)
        mlflow.log_param("clean_filename", clean_filename)
        mlflow.log_param("num_rows_loaded", before)
        mlflow.log_param("num_rows_after_dropna", after)
        mlflow.log_param("num_cat_features", len(categorical_features))
        mlflow.log_param("num_num_features", len(numeric_features))

        # Log human-readable pipeline parameters as YAML
        pipeline_params = {
            "raw_filename": raw_filename,
            "clean_filename": clean_filename,
            "num_rows_loaded": before,
            "num_rows_after_dropna": after,
            "num_cat_features": len(categorical_features),
            "num_num_features": len(numeric_features),
            "categorical_columns": categorical_features,
            "numerical_columns": numeric_features,
        }
        params_yaml_path = f"{domino_artifact_dir}/preprocessing_pipeline_params.yaml"
        os.makedirs(os.path.dirname(params_yaml_path), exist_ok=True)
        with open(params_yaml_path, "w") as f:
            yaml.dump(pipeline_params, f, default_flow_style=False)
        mlflow.log_artifact(params_yaml_path, artifact_path="params")

        # Log the pipeline as a single model
        X_sig = X.copy()
        for col in numeric_features:
            if np.issubdtype(X_sig[col].dtype, np.integer):
                X_sig[col] = X_sig[col].astype("float64")
        signature = infer_signature(X_sig.iloc[:5], pipeline.transform(X_sig.iloc[:5]))
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="preprocessing_pipeline",
            registered_model_name="CC Fraud Preprocessing",
            signature=signature
        )
        mlflow.set_tag("pipeline", "full_preproc_no_pca")

        # Log metrics
        mlflow.log_metric("pct_data_removed", pct_removed)
        mlflow.log_metric("num_rows_removed", before - after)
        mlflow.log_metric("preproc_fit_time_sec", fit_time)

        # 7) Generate and log artifacts (corr, scatter, etc.)
        num_df = df.select_dtypes(include="number").drop(columns=["Time", "Class"], errors="ignore")
        # Correlation heatmap
        plt.figure(figsize=(14,12))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="vlag")
        plt.title("Correlation Matrix")
        corr_path = f"{domino_artifact_dir}/raw_correlation_matrix.png"
        os.makedirs(os.path.dirname(corr_path), exist_ok=True)
        plt.savefig(corr_path); plt.close()
        mlflow.log_artifact(corr_path, artifact_path="plots")
        # Scatter matrix
        sample_df = num_df.sample(n=500, random_state=0)
        fig = scatter_matrix(sample_df, alpha=0.2, diagonal="hist", figsize=(15,15))
        scatter_path = f"{domino_artifact_dir}/raw_scatter_plots.png"
        os.makedirs(os.path.dirname(scatter_path), exist_ok=True)
        plt.savefig(scatter_path); plt.close()
        mlflow.log_artifact(scatter_path, artifact_path="plots")

        # 8) EDA HTML
        profile = ProfileReport(df, title="EDA Report", explorative=True, minimal=True)
        eda_path = f"{domino_artifact_dir}/preprocessing_report.html"
        os.makedirs(os.path.dirname(eda_path), exist_ok=True)
        profile.to_file(eda_path)
        mlflow.log_artifact(eda_path, artifact_path="eda")

    # Return only file paths for Flyte compatibility
    return features_path, labels_path, clean_path

if __name__ == "__main__":

    raw_filename="raw_cc_transactions.csv",
    clean_filename="preprocessing_processed_cc_transactions.csv"
    experiment_name = f"CC Fraud Preprocessing {domino_short_id()}"

    raw_df, features_processed, y = run_data_ingestion_and_processing(
        raw_filename=raw_filename,
        clean_filename=clean_filename,
        experiment_name=experiment_name
    )