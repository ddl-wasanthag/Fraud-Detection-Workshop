import io, os, time

import pandas as pd
import numpy as np
from domino_data.data_sources import DataSourceClient
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from helpers.generation_labels import get_generation

clean_filename = 'clean_cc_transactions.csv'
features_filename = 'transformed_cc_transactions.csv'
labels_filename = 'credit_card_fraud_detection'
domino_working_dir = os.environ.get("DOMINO_WORKING_DIR", ".")
domino_project_name = os.environ.get("DOMINO_PROJECT_NAME", "my-local-project")

# read clean dataset
domino_dataset_dir = f"{domino_working_dir.replace('code', 'data')}/{domino_project_name}"
domino_artifact_dir = domino_working_dir.replace('code', 'artifacts')

clean_path = f"{domino_dataset_dir}/{clean_filename}"
clean_df = pd.read_csv(clean_path)
print(f"Loaded {len(clean_df):,} rows from {clean_path}")
print(clean_df.columns)

# generate derived columns
def add_derived_features(df):
    # Velocity features
    df['amount_vs_avg30d_ratio'] = df['Amount'] / (df['Avg30d'] + 1e-6)

    # Risk composite scores
    df['risk_score'] = (df['MerchantRisk'] + df['IPReputation']) / 2
    df['trust_score'] = df['DeviceTrust'] - df['MerchantRisk']
    
    df['generation'] = df['Age'].apply(get_generation)
    
    return df

full_cleaned_df = add_derived_features(clean_df)

# separate labels from features
labels_df = full_cleaned_df['Class']
features_df = full_cleaned_df.drop(columns=['Class'], errors='ignore')
numeric_features_df = features_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features_df = features_df.select_dtypes(include=[object, "category"]).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features_df),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_df)
    ]
)

pipeline = Pipeline([("preproc", preprocessor)])

start_time = time.time()
transformed_features_array = pipeline.fit_transform(features_df)
fit_time = time.time() - start_time
print(f"Transformed features in {fit_time:.2} seconds.")


# Convert back to DataFrame with proper column names
feature_names = pipeline.named_steps['preproc'].get_feature_names_out()
transformed_features_df = pd.DataFrame(transformed_features_array, columns=feature_names, index=features_df.index)

# Add labels back
transformed_features_df['Class'] = labels_df

# Save transformed features and labels
transformed_features_df.to_csv(f"{domino_dataset_dir}/{features_filename}", index=False)
print('saved to ', f"{domino_dataset_dir}/{clean_filename}")
