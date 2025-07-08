from sklearn.ensemble import AdaBoostClassifier

from exercises.c_DevelopmentAndTraining.generic_trainer import train_fraud

# Load DataFrame from dataset
transformed_df_filename = 'transformed_cc_transactions.csv'

model_name = 'AdaBoost'
model_obj = AdaBoostClassifier(
            n_estimators=10,
            learning_rate=0.1,
            algorithm="SAMME",
        )

for i in range(50):
    res = train_fraud(model_obj, model_name, transformed_df_filename)

    print(f"Training {model_name} completed successfully")
    print(res)

