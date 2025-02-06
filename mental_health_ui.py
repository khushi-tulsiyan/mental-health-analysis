import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict

def load_model_and_preprocessor():
    model = joblib.load('./models/saved/best_model.joblib')
    preprocessor = joblib.load('./models/saved/preprocessor.joblib')
    return model, preprocessor

def get_phq9_questions() -> Dict[str, str]:

    return {
        'q1': 'Little interest or pleasure in doing things',
        'q2': 'Feeling down, depressed, or hopeless',
        'q3': 'Trouble falling/staying asleep, sleeping too much',
        'q4': 'Feeling tired or having little energy',
        'q5': 'Poor appetite or overeating',
        'q6': 'Feeling bad about yourself/failure',
        'q7': 'Trouble concentrating on things',
        'q8': 'Moving/speaking slowly or being fidgety',
        'q9': 'Thoughts of self-harm'
    }

def get_response_options() -> Dict[int, str]:

    return {
        0: 'Not at all',
        1: 'Several days',
        2: 'More than half the days',
        3: 'Nearly every day'
    }

def create_phq9_input() -> Dict[str, int]:
    
    st.subheader("Over the last 2 weeks, how often have you been bothered by any of the following problems?")
    
    responses = {}
    questions = get_phq9_questions()
    options = get_response_options()
    
    for q_id, question in questions.items():
        responses[q_id] = st.radio(
            question,
            options=list(options.keys()),
            format_func=lambda x: options[x],
            horizontal=True,
            key=q_id
        )
        st.write("---")
    
    return responses

def display_severity_gauge(total_score: int):
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = total_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 27], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 4], 'color': "lightgreen"},
                {'range': [5, 9], 'color': "yellow"},
                {'range': [10, 14], 'color': "orange"},
                {'range': [15, 19], 'color': "salmon"},
                {'range': [20, 27], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': total_score
            }
        },
        title = {'text': "Depression Severity Score"}
    ))
    st.plotly_chart(fig)

def main():
    st.title("PHQ-9 Depression Screening Tool")
    
    st.write("""
    This tool helps assess depression severity using the PHQ-9 questionnaire.
    Please answer all questions honestly based on your experiences over the past 2 weeks.
    """)
    
    try:
        model, preprocessor = load_model_and_preprocessor()
        
        responses = create_phq9_input()
        
        if st.button("Analyze Responses"):
            df = pd.DataFrame([responses])
        
            processed_input, _ = preprocessor.prepare_data(df)
            
        
            total_score = sum(responses.values())
            
            
            display_severity_gauge(total_score)
            
            
            prediction = model.predict(processed_input)
            probabilities = model.predict_proba(processed_input)
            
            
            st.subheader("Analysis Results")
            
            severity_level = preprocessor.calculate_severity_level(total_score)
            st.write(f"Based on your responses, your depression severity level is: **{severity_level.replace('_', ' ').title()}**")
            
            
            st.subheader("Symptom Breakdown")
            fig = px.bar(
                x=list(get_phq9_questions().values()),
                y=list(responses.values()),
                labels={'x': 'Symptoms', 'y': 'Severity'},
                title="Symptom Severity Distribution"
            )
            st.plotly_chart(fig)
            
            
            st.subheader("Recommendations")
            if total_score >= 10:
                st.warning("""
                Your responses indicate significant depression symptoms. 
                It is strongly recommended that you consult with a mental health professional.
                """)
            if total_score >= 20:
                st.error("""
                Your responses indicate severe depression symptoms. 
                Please seek immediate professional help or contact a crisis hotline.
                """)
            
        
            if responses['q9'] >= 2:
                st.error("""
                **IMPORTANT**: If you're having thoughts of self-harm or suicide, 
                please seek immediate help:
                - National Suicide Prevention Lifeline: 988
                - Crisis Text Line: Text HOME to 741741
                """)
            
            st.info("""
            Remember: This tool is for screening purposes only and does not replace 
            professional medical advice, diagnosis, or treatment.
            """)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure the model and preprocessor files are properly saved.")

if __name__ == "__main__":
    main()