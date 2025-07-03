# File: exercises/b_DataProcessingAndExploration/preprocessing.py

"""
Data Ingestion, Processing, and MLflow Model Logging Pipeline

This module handles the complete data preprocessing pipeline for credit card fraud detection,
including data loading, cleaning, feature engineering, and MLflow logging.
"""

import io
import os
import time
import yaml
import shutil
from pathlib import Path
from typing import Tuple, Optional

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


class DataPreprocessor:
    """Handles data preprocessing pipeline for credit card fraud detection."""
    
    def __init__(self):
        self.domino_working_dir = os.environ.get("DOMINO_WORKING_DIR", ".")
        self.domino_datasource_dir = self.domino_working_dir.replace('code', 'data')
        self.domino_artifact_dir = self.domino_working_dir.replace('code', 'artifacts')
        self.domino_project_name = os.environ.get("DOMINO_PROJECT_NAME", "my-local-project")
        
        # Ensure directories exist
        self.output_dir = Path(f"{self.domino_datasource_dir}/{self.domino_project_name}")
        self.artifact_dir = Path(self.domino_artifact_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        
        self.pipeline = None
        self.numeric_features = []
        self.categorical_features = []
    
    def load_data(self, raw_filename: str) -> pd.DataFrame:
        """Load data from Domino data source."""
        # Handle case where raw_filename might be a tuple/list
        if isinstance(raw_filename, (list, tuple)):
            if len(raw_filename) == 0:
                raise ValueError("raw_filename must not be empty")
            raw_filename = raw_filename[0]
        
        try:
            ds = DataSourceClient().get_datasource("credit_card_fraud_detection")
            buf = io.BytesIO()
            ds.download_fileobj(raw_filename, buf)
            buf.seek(0)
            df = pd.read_csv(buf)
            print(f"🔍 Loaded {len(df):,} rows from {raw_filename}")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {raw_filename}: {str(e)}")
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Clean the dataset and return cleaning statistics."""
        before = len(df)
        df_clean = df.dropna()
        after = len(df_clean)
        pct_removed = 100 * (before - after) / before if before > 0 else 0
        
        cleaning_stats = {
            'rows_before': before,
            'rows_after': after,
            'rows_removed': before - after,
            'pct_removed': pct_removed
        }
        
        print(f"🧹 Dropped {before - after:,} rows with missing data ({pct_removed:.2f}%)")
        return df_clean, cleaning_stats
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable."""
        # Drop target and time-related columns from features
        feature_cols = df.drop(columns=["Class", "Time", "Hour"], errors="ignore")
        target = df["Class"]
        
        # Identify feature types
        self.numeric_features = feature_cols.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = feature_cols.select_dtypes(include=[object, "category"]).columns.tolist()
        
        print(f"📊 Features: {len(self.numeric_features)} numeric, {len(self.categorical_features)} categorical")
        return feature_cols, target
    
    def create_preprocessing_pipeline(self) -> Pipeline:
        """Create and return the preprocessing pipeline."""
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)
            ]
        )
        
        self.pipeline = Pipeline([("preproc", preprocessor)])
        return self.pipeline
    
    def fit_transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Fit the pipeline and transform features."""
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call create_preprocessing_pipeline() first.")
        
        start_time = time.time()
        features_processed = self.pipeline.fit_transform(X)
        fit_time = time.time() - start_time
        
        print(f"⚡ Pipeline fit/transform completed in {fit_time:.2f} seconds")
        return features_processed, fit_time
    
    def save_processed_data(self, features_processed: np.ndarray, y: pd.Series, 
                          clean_filename: str) -> Tuple[str, str, str]:
        """Save processed data to files."""
        # Save numpy array and labels
        features_path = str(self.output_dir / "preprocessing_features_processedd.npy")
        labels_path = str(self.output_dir / "preprocessing_feature_labelss.csv")
        
        np.save(features_path, features_processed)
        y.to_csv(labels_path, index=False)
        
        # Create DataFrame with processed features
        if hasattr(features_processed, "toarray"):
            X_arr = features_processed.toarray()
        else:
            X_arr = features_processed
        
        # Get feature names
        num_cols = self.numeric_features
        cat_cols = (self.pipeline.named_steps["preproc"]
                   .named_transformers_["cat"]
                   .get_feature_names_out(self.categorical_features).tolist())
        all_cols = num_cols + cat_cols
        
        # Create final DataFrame
        df_scaled = pd.DataFrame(X_arr, columns=all_cols)
        df_scaled["Class"] = y.values
        
        # Save cleaned DataFrame
        clean_path = str(self.output_dir / clean_filename)
        df_scaled.to_csv(clean_path, index=False)
        
        print(f"✅ Saved processed data:")
        print(f"   - Features: {features_path}")
        print(f"   - Labels: {labels_path}")
        print(f"   - Clean data: {clean_path}")

        clean_df = df_scaled.copy()

        return clean_df, clean_path, features_path, labels_path
    
    def generate_visualizations(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Generate and save visualization artifacts."""
        # Prepare numeric data for visualization
        num_df = df.select_dtypes(include="number").drop(columns=["Time", "Class"], errors="ignore")
        
        # Correlation heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="vlag")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        
        corr_path = str(self.artifact_dir / "raw_correlation_matrix.png")
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Scatter matrix (sample for performance)
        sample_size = min(500, len(num_df))
        sample_df = num_df.sample(n=sample_size, random_state=42)
        
        fig = scatter_matrix(sample_df, alpha=0.2, diagonal="hist", figsize=(15, 15))
        scatter_path = str(self.artifact_dir / "raw_scatter_plots.png")
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Generated visualizations:")
        print(f"   - Correlation matrix: {corr_path}")
        print(f"   - Scatter plots: {scatter_path}")
        
        return corr_path, scatter_path
    
    def generate_eda_report(self, df: pd.DataFrame) -> str:
        """Generate comprehensive EDA report."""
        profile = ProfileReport(
            df, 
            title="Credit Card Fraud Detection - EDA Report",
            explorative=True,
            minimal=True
        )
        
        eda_path = str(self.artifact_dir / "preprocessing_report.html")
        profile.to_file(eda_path)
        
        print(f"📄 Generated EDA report: {eda_path}")
        return eda_path
    
    def log_to_mlflow(self, raw_filename: str, clean_filename: str, 
                     cleaning_stats: dict, fit_time: float,
                     clean_path: str, corr_path: str, scatter_path: str, 
                     eda_path: str, X_sample: pd.DataFrame) -> None:
        """Log all artifacts and metrics to MLflow."""
        
        # Log artifacts
        mlflow.log_artifact(clean_path, artifact_path="data")
        mlflow.log_artifact(corr_path, artifact_path="plots")
        mlflow.log_artifact(scatter_path, artifact_path="plots")
        mlflow.log_artifact(eda_path, artifact_path="eda")
        
        # Log parameters
        mlflow.log_param("raw_filename", raw_filename)
        mlflow.log_param("clean_filename", clean_filename)
        mlflow.log_param("num_rows_loaded", cleaning_stats['rows_before'])
        mlflow.log_param("num_rows_after_dropna", cleaning_stats['rows_after'])
        mlflow.log_param("num_cat_features", len(self.categorical_features))
        mlflow.log_param("num_num_features", len(self.numeric_features))
        
        # Log metrics
        mlflow.log_metric("pct_data_removed", cleaning_stats['pct_removed'])
        mlflow.log_metric("num_rows_removed", cleaning_stats['rows_removed'])
        mlflow.log_metric("preproc_fit_time_sec", fit_time)
        
        # Log pipeline parameters as YAML
        pipeline_params = {
            "raw_filename": raw_filename,
            "clean_filename": clean_filename,
            "preprocessing_steps": ["StandardScaler", "OneHotEncoder"],
            "categorical_columns": self.categorical_features,
            "numerical_columns": self.numeric_features,
            **cleaning_stats
        }
        
        params_yaml_path = str(self.artifact_dir / "preprocessing_pipeline_params.yaml")
        with open(params_yaml_path, "w") as f:
            yaml.dump(pipeline_params, f, default_flow_style=False)
        mlflow.log_artifact(params_yaml_path, artifact_path="params")
        
        # Log the preprocessing pipeline model
        X_sig = X_sample.copy()
        for col in self.numeric_features:
            if np.issubdtype(X_sig[col].dtype, np.integer):
                X_sig[col] = X_sig[col].astype("float64")
        
        signature = infer_signature(X_sig.iloc[:5], self.pipeline.transform(X_sig.iloc[:5]))
        mlflow.sklearn.log_model(
            self.pipeline,
            artifact_path="preprocessing_pipeline",
            registered_model_name="CC Fraud Preprocessing",
            signature=signature
        )
        mlflow.set_tag("pipeline", "full_preproc_no_pca")
        
        print("📝 Logged all artifacts to MLflow")
    
    def handle_flow_output(self, clean_path: str) -> None:
        """Handle output for Domino Flow if running as a workflow job."""
        if os.environ.get("DOMINO_IS_WORKFLOW_JOB", "false").lower() == "true":
            flow_output_path = "/workflow/outputs/processed_data_path"
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(flow_output_path), exist_ok=True)
            
            # Copy file to expected location
            shutil.copyfile(clean_path, flow_output_path)
            
            # Verify file was created successfully
            if not os.path.exists(flow_output_path):
                raise FileNotFoundError(f"Expected flow output file not found: {flow_output_path}")
            
            print(f"🔗 Flow output written: {flow_output_path}")
            print(f"   File size: {os.path.getsize(flow_output_path):,} bytes")


def run_data_ingestion_and_processing(raw_filename: str, clean_filename: str, 
                                    experiment_name: str) -> Tuple[str, str, str]:
    """
    Main function to run the complete data preprocessing pipeline.
    
    Args:
        raw_filename: Name of the raw data file
        clean_filename: Name for the cleaned output file
        experiment_name: MLflow experiment name
        
    Returns:
        Tuple of (features_path, labels_path, clean_path)
    """
    # Initialize MLflow
    mlflow.set_experiment(experiment_name)
    
    # End any existing run
    if mlflow.active_run():
        mlflow.end_run()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    with mlflow.start_run(run_name="Preprocessing Pipeline") as run:
        # Load and clean data
        df = preprocessor.load_data(raw_filename).head(102)
        df_clean, cleaning_stats = preprocessor.clean_data(df)
        
        # Prepare features
        X, y = preprocessor.prepare_features(df_clean)
        
        # Create and fit preprocessing pipeline
        preprocessor.create_preprocessing_pipeline()
        features_processed, fit_time = preprocessor.fit_transform_features(X)
        
        # Save processed data
        clean_df, clean_path, features_path, labels_path = preprocessor.save_processed_data(
            features_processed, y, clean_filename
        )
        
        # Generate visualizations and reports
        corr_path, scatter_path = preprocessor.generate_visualizations(df_clean)
        eda_path = preprocessor.generate_eda_report(df_clean)
        
        # Log everything to MLflow
        preprocessor.log_to_mlflow(
            raw_filename, clean_filename, cleaning_stats, fit_time,
            clean_path, corr_path, scatter_path, eda_path, X
        )
        
        # Handle flow output if needed
        preprocessor.handle_flow_output(clean_path)
        
        print("✅ Pipeline completed successfully!")
        return clean_df, clean_path, features_path, labels_path


if __name__ == "__main__":
    # Configuration
    raw_filename = "raw_cc_transactions.csv"  # Fixed: removed trailing comma
    clean_filename = "preprocessing_processed_cc_transactions.csv"
    experiment_name = f"CC Fraud Preprocessing {domino_short_id()}"
    
    # Run the pipeline
    try:
        clean_df, clean_path, features_path, labels_path = run_data_ingestion_and_processing(
            raw_filename=raw_filename,
            clean_filename=clean_filename,
            experiment_name=experiment_name
        )
        print(f"\n🎉 Processing complete!")
        print(f"   Features: {features_path}")
        print(f"   Labels: {labels_path}")
        print(f"   Clean data: {clean_path}")
        print(f"   Clean data (head): {clean_df.head()}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        raise