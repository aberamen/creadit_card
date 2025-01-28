import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib

# Load preprocessed data
data = pd.read_csv('src/data/processed/task3_output.csv')

# Splitting the data
X = data.drop(columns=['default'])
y = data['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Models
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(random_state=42)

# Train Models
log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Evaluate Models
y_pred_log = log_reg.predict(X_test)
y_pred_rf = rf.predict(X_test)

print("Logistic Regression Performance:")
print(classification_report(y_test, y_pred_log))

print("Random Forest Performance:")
print(classification_report(y_test, y_pred_rf))

# Save Models
joblib.dump(rf, 'src/models/saved_model.pkl')
print("Random Forest Model Saved Successfully!")
