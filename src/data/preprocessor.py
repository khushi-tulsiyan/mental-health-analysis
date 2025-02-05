import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


file_path = "./data/raw/Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv"
df = pd.read_csv(file_path)


drop_cols = ['Unnamed: 0', 'user_id', 'time', 'start.time', 'id', 'period.name']
df.drop(columns=drop_cols, inplace=True)


for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].median(), inplace=True)


df['sex'].fillna('unknown', inplace=True)
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])


scaler = StandardScaler()
numeric_cols = ['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9', 'age', 'happiness.score']

numeric_cols = [col for col in numeric_cols if col in df.columns]
if numeric_cols:
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


def classify_phq9(score):
    if score <= 4:
        return 'Minimal'
    elif score <= 9:
        return 'Mild'
    elif score <= 14:
        return 'Moderate'
    elif score <= 19:
        return 'Moderately Severe'
    else:
        return 'Severe'

df['depression_severity'] = df[['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9']].sum(axis=1).apply(classify_phq9)


df['depression_severity'] = LabelEncoder().fit_transform(df['depression_severity'])


X = df.drop(columns=['depression_severity'])
y = df['depression_severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.to_csv("./data/processed/X_train.csv", index=False)
X_test.to_csv("./data/processed/X_test.csv", index=False)
y_train.to_csv("./data/processed/y_train.csv", index=False)
y_test.to_csv("./data/processed/y_test.csv", index=False)
