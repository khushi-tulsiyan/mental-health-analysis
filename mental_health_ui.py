import streamlit as st
import joblib
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the best trained model
best_model = joblib.load("./models/saved/best_model.pkl")

# Load the tokenizer and Llama 2 model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Define function to get model prediction
def predict_depression(phq_scores):
    df = pd.DataFrame([phq_scores], columns=[f"phq{i+1}" for i in range(9)])
    prediction = best_model.predict(df)[0]
    return prediction

# Define function to generate explanation
def generate_llama_explanation(prediction):
    prompt = f"A user has been classified as having {prediction} depression severity. Provide a brief explanation and coping mechanisms."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = llama_model.generate(input_ids, max_length=150)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit UI
st.title("Mental Health Assessment - PHQ-9")
st.write("Enter your PHQ-9 scores below to assess your depression severity.")

# User input for PHQ-9 scores
phq_scores = []
for i in range(9):
    phq_scores.append(st.slider(f"PHQ-{i+1} Score (0-3)", 0, 3, 0))

# Predict button
if st.button("Predict Mental Health Condition"):
    prediction = predict_depression(phq_scores)
    explanation = generate_llama_explanation(prediction)
    
    st.subheader("Prediction:")
    st.write(f"**{prediction} Depression**")
    
    st.subheader("Explanation & Coping Strategies:")
    st.write(explanation)
