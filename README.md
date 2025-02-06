# Mental Health Analysis and Severity Predictor

This project implements a machine learning system for analyzing PHQ-9 (Patient Health Questionnaire-9) responses and predicting depression severity levels. It includes data preprocessing, model training, and both UI and command-line interfaces for making predictions.

## About PHQ-9

The PHQ-9 is a validated screening tool used to assess depression severity. It consists of 9 questions that correspond to the DSM-IV criteria for major depressive disorder. Each question is scored from 0 (not at all) to 3 (nearly every day), with total scores ranging from 0 to 27.

Depression Severity Levels:
- 0-4: Minimal depression
- 5-9: Mild depression
- 10-14: Moderate depression
- 15-19: Moderately severe depression
- 20-27: Severe depression

## Setup

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
mental_health_analysis/
├── data/
│   ├── raw/                  # Store raw PHQ-9 dataset
│   └── processed/            # Store preprocessed data
├── models/
│   ├── saved/               # Trained models
src
│──── data/               # Data processing modules
│──── models/             # Model training modules
│──── utils/              # Helper functions
|────llm/
ui/                     # Streamlit interface
```

## Usage

### Training the Model

1. Run the training script:
```bash
python src/models/model_trainer.py
```

### Using the Streamlit Interface

1. Start the Streamlit app:
```bash
streamlit run mental_health_ui.py
```

2. Access the UI in your browser at `http://localhost:8501`



## Model Performance

The system evaluates models using:
- Accuracy
- Precision
- Recall
- F1-score
- Cohen's Kappa (for ordinal classification)
- Mean severity difference
- Severe case recall


## Input Format

The PHQ-9 questions assess the following symptoms over the last 2 weeks:

1. q1: Little interest or pleasure in doing things
2. q2: Feeling down, depressed, or hopeless
3. q3: Trouble falling/staying asleep, sleeping too much
4. q4: Feeling tired or having little energy
5. q5: Poor appetite or overeating
6. q6: Feeling bad about yourself/failure
7. q7: Trouble concentrating on things
8. q8: Moving/speaking slowly or being fidgety
9. q9: Thoughts of self-harm

Each question should be scored:
- 0: Not at all
- 1: Several days
- 2: More than half the days
- 3: Nearly every day

## Important Notes

- This tool is for screening purposes only and does not replace professional medical diagnosis
- If you're experiencing thoughts of self-harm (q9 ≥ 2), please seek immediate professional help
- The model provides confidence scores and uncertainty estimates for transparency
- SHAP values are used to explain which symptoms contributed most to the prediction

## Crisis Resources

- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See LICENSE file for details