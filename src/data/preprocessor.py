import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any

class PHQ9Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def calculate_severity_level(self, total_score: int) -> str:
        
        if total_score <= 4:
            return 'minimal'
        elif total_score <= 9:
            return 'mild'
        elif total_score <= 14:
            return 'moderate'
        elif total_score <= 19:
            return 'moderately_severe'
        else:
            return 'severe'
    
    def validate_phq9_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        
        question_cols = [col for col in df.columns if col.startswith('q')]
        for col in question_cols:
            if df[col].max() > 3 or df[col].min() < 0:
                raise ValueError(f"Invalid scores found in {col}. Scores must be between 0 and 3.")
        return df
    
    def calculate_total_score(self, df: pd.DataFrame) -> pd.DataFrame:
        
        question_cols = [col for col in df.columns if col.startswith('q')]
        df['total_score'] = df[question_cols].sum(axis=1)
        df['severity_level'] = df['total_score'].apply(self.calculate_severity_level)
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df['mood_score'] = df[['q1', 'q2']].mean(axis=1) 
        df['physical_score'] = df[['q3', 'q4', 'q5']].mean(axis=1)  
        df['cognitive_score'] = df[['q6', 'q7']].mean(axis=1)
        df['behavioral_score'] = df[['q8']].mean(axis=1)  
        df['risk_score'] = df[['q9']].mean(axis=1)  
        
        
        question_cols = [col for col in df.columns if col.startswith('q')]
        df['symptom_variability'] = df[question_cols].std(axis=1)
        
        
        df['has_severe_symptoms'] = (df[question_cols] == 3).any(axis=1).astype(int)
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'severity_level') -> Tuple[pd.DataFrame, pd.Series]:
              
        df = self.validate_phq9_scores(df)
            
        df = self.calculate_total_score(df)  
        
        df = self.engineer_features(df)

        df = pd.DataFrame(
            self.imputer.fit_transform(df),
            columns=df.columns,
            index=df.index
        )
        
        if target_col:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df
            
        feature_cols = [col for col in X.columns if col not in ['total_score', 'severity_level']]
        X[feature_cols] = self.scaler.fit_transform(X[feature_cols])
        
        return X, y
    
    def get_feature_names(self) -> Dict[str, Any]:
        
        return {
            'questions': {
                'q1': 'Little interest or pleasure',
                'q2': 'Feeling down or hopeless',
                'q3': 'Sleep issues',
                'q4': 'Feeling tired',
                'q5': 'Appetite problems',
                'q6': 'Feeling bad about yourself',
                'q7': 'Concentration problems',
                'q8': 'Moving/speaking slowly or being fidgety',
                'q9': 'Thoughts of self-harm'
            },
            'derived_features': {
                'mood_score': 'Average of core mood symptoms',
                'physical_score': 'Average of physical symptoms',
                'cognitive_score': 'Average of cognitive symptoms',
                'behavioral_score': 'Behavioral symptoms score',
                'risk_score': 'Self-harm risk score',
                'symptom_variability': 'Variability across symptoms',
                'has_severe_symptoms': 'Presence of any severe symptoms'
            }
        }