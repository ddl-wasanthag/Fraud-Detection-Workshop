from sklearn.naive_bayes import GaussianNB

from exercises.c_DevelopmentAndTraining.generic_trainer import train_fraud

# Load DataFrame from dataset
transformed_df_filename = 'transformed_cc_transactions.csv'

model_name = 'GaussianNB'
model_obj = GaussianNB()

for i in range(30):
    res = train_fraud(model_obj, model_name, transformed_df_filename)

    print(f"Training {model_name} completed successfully")
    print(res)

