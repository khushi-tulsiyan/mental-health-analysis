import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import RandomOverSampler
import joblib
import shap
import matplotlib.pyplot as plt


train_data = pd.read_csv("data/train_data.csv")
test_data = pd.read_csv("data/test_data.csv")


X_train, y_train = train_data.drop(columns=['treatment']), train_data['treatment']
X_test, y_test = test_data.drop(columns=['treatment']), test_data['treatment']


oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)


xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)



THRESHOLD = 0.0004 

def custom_predict(model, X_test):
    probabilities = model.predict_proba(X_test)  # Get class probabilities
    class_1_prob = probabilities[:, 1]  # Probability of "Needs Treatment"
    
    # Predict Class 1 if probability > threshold, else Class 0
    predictions = (class_1_prob >= THRESHOLD).astype(int)
    return predictions


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = custom_predict(model, X_test)  # Use custom threshold
    y_prob = model.predict_proba(X_test)  # Get probability scores

    print(f"\n{model_name} Performance:")
    print(classification_report(y_test, y_pred, zero_division=1))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=1))
    print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=1))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))


X_test = pd.DataFrame(X_test, columns=train_data.drop(columns=['treatment']).columns)




evaluate_model(rf_model, X_test, y_test, "Random Forest")
evaluate_model(xgb_model, X_test, y_test, "XGBoost")


explainer = shap.Explainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


joblib.dump(xgb_model, "models/mental_health_model.pkl")
print("Model saved successfully!")
