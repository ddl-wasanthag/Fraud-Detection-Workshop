# File: feature_engineering.py
import io, os, time
from pathlib import Path

import pandas as pd
import numpy as np
from domino_data.data_sources import DataSourceClient
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
from mlflow.models import infer_signature
from ydata_profiling import ProfileReport

from helpers.generation_labels import get_generation
from helpers.domino_short_id import domino_short_id

experiment_name = f"CC Fraud Preprocessing {domino_short_id()}"
clean_filename = 'clean_cc_transactions.csv'
features_filename = 'transformed_cc_transactions.csv'
domino_working_dir = os.environ.get("DOMINO_WORKING_DIR", ".")
domino_project_name = os.environ.get("DOMINO_PROJECT_NAME", "my-local-project")

domino_dataset_dir = f"{domino_working_dir.replace('code', 'data')}/{domino_project_name}"
domino_artifact_dir = domino_working_dir.replace('code', 'artifacts')
clean_path = f"{domino_dataset_dir}/{clean_filename}"

def add_derived_features(df):
    # Velocity features
    df['amount_vs_avg30d_ratio'] = df['Amount'] / (df['Avg30d'] + 1e-6)

    # Risk composite scores
    df['risk_score'] = (df['MerchantRisk'] + df['IPReputation']) / 2
    df['trust_score'] = df['DeviceTrust'] - df['MerchantRisk']
    
    df['generation'] = df['Age'].apply(get_generation)
    
    return df

mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="Preprocessing Pipeline") as run:
    # read clean dataset
    clean_df = pd.read_csv(clean_path, index_col=0)
    print(f"Loaded {len(clean_df):,} rows from {clean_path}")
    print(clean_df.columns)
    
    # generate derived features
    full_cleaned_df = add_derived_features(clean_df)
    
    # separate labels from features
    labels_df = full_cleaned_df['Class']
    features_df = full_cleaned_df.drop(columns=['Class'], errors='ignore')
    numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features_df.select_dtypes(include=[object,
                                                                "category"]).columns.tolist()
    print(features_df)
    # create and run preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             categorical_features)
        ]
    )
    pipeline = Pipeline([("preproc", preprocessor)])
    start_time = time.time()
    transformed_features_array = pipeline.fit_transform(features_df)
    fit_time = time.time() - start_time
    print(f"Transformed features in {fit_time:.2} seconds.")

    # Convert back to DataFrame with proper column names
    feature_names = pipeline.named_steps['preproc'].get_feature_names_out()
    transformed_features_df = pd.DataFrame(transformed_features_array,
                                           columns=feature_names, index=features_df.index)

    # Add labels back
    transformed_features_df['Class'] = labels_df
    
    # Save transformed features and labels
    transformed_features_df.to_csv(f"{domino_dataset_dir}/{features_filename}", index=False)
    print('saved to ', f"{domino_dataset_dir}/{features_filename}")

    # Save EDA report
    profile = ProfileReport(
        clean_df, 
        title="Credit Card Fraud Detection - EDA Report",
        explorative=True,
        minimal=True
    )
    
    eda_path = f"{domino_artifact_dir}/preprocessing_report.html"
    profile.to_file(eda_path)

    # Log everything to MLflow
    mlflow.log_artifact(clean_path, artifact_path="data")
    mlflow.log_artifact(eda_path, artifact_path="eda")
    mlflow.log_param("num_rows_loaded", len(features_df))
    mlflow.log_param("num_cat_features", len(categorical_features))
    mlflow.log_param("num_num_features", len(numeric_features))
    mlflow.log_metric("fit_time", fit_time)

    # Add predict method to pipeline (alias for transform)
    pipeline.predict = pipeline.transform
    
    # Log the preprocessing pipeline model
    X_sample = features_df.iloc[:20].copy()
    for col in numeric_features:
        if np.issubdtype(X_sample[col].dtype, np.integer):
            X_sample[col] = X_sample[col].astype("float64")
    
    y_sample = pipeline.transform(X_sample)
    signature = infer_signature(X_sample, y_sample)
    
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="preprocessing_pipeline",
        # registered_model_name="CC Fraud Preprocessing",  Use if you want this to automatically register as a Domino model.
        signature=signature
    )
    mlflow.set_tag("pipeline", "preprocessing")

    Path("/workflow/outputs/transformed_filename").write_text(features_filename)
