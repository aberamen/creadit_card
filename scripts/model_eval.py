import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# Load the test data
data = pd.read_csv('src/data/processed/task3_output.csv')
X_test = data.drop(columns=['default'])
y_test = data['default']

# Load the model
model = joblib.load('src/models/saved_model.pkl')

# Evaluate Model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Model Performance:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.2f}")
