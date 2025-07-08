from xgboost import XGBClassifier

from exercises.c_DevelopmentAndTraining.generic_trainer import train_fraud

# Load DataFrame from dataset
transformed_df_filename = 'transformed_cc_transactions.csv'

model_name = 'XGBoost'
model_obj = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="auc",
        )

res = train_fraud(model_obj, model_name, transformed_df_filename)

print(f"Training {model_name} completed successfully")
print(res)

