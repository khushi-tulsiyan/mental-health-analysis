import joblib
import pandas as pd
import numpy as np
import streamlit as st


model = joblib.load("models/mental_health_model.pkl")


feature_names = ['family_history', 'work_interfere', 'mental_vs_physical', 'coworkers', 'leave', 'benefits']


def main():
    st.title("Mental Health Prediction Tool")
    st.write("Enter your details below to check for potential mental health conditions.")
    
    
    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.selectbox(f"{feature.replace('_', ' ').title()}", ['Yes', 'No'])
    
    if st.button("Predict"):
        
        input_data = {k: 1 if v == 'Yes' else 0 for k, v in input_data.items()}
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0]  # Get probability scores
        
        condition = "Likely to need mental health treatment" if prediction == 1 else "Unlikely to need treatment"
        st.success(f"Prediction: {condition}")
        
        
        st.write("Prediction Probabilities:")
        for i, prob in enumerate(prediction_proba):
            st.write(f"Class {i}: {prob:.4f}")

if __name__ == "__main__":
    main()
