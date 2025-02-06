import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, recall_score
from sklearn.metrics import classification_report, cohen_kappa_score
import shap
import joblib
from typing import Dict, Any, List, Tuple

class PHQ9ModelTrainer:
    def __init__(self):
        """Initialize models specifically tuned for PHQ-9 depression severity prediction."""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                min_samples_leaf=7,
                random_state=42
            )
        }
        self.best_model = None
        self.best_model_name = None
        self.severity_levels = ['minimal', 'mild', 'moderate', 'moderately_severe', 'severe']
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Evaluate model with metrics specific to depression severity assessment."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Calculate Cohen's Kappa (important for ordinal classifications)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Calculate severity difference (how far off predictions are)
        severity_diff = np.abs(
            np.array([self.severity_levels.index(y) for y in y_true]) - 
            np.array([self.severity_levels.index(y) for y in y_pred])
        )
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'kappa': kappa,
            'mean_severity_difference': severity_diff.mean(),
            'severe_case_recall': recall_score(
                y_true == 'severe',
                y_pred == 'severe'
            )
        }
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """Train and evaluate models using cross-validation."""
        results = {}
        best_kappa = 0
        
        for name, model in self.models.items():
            # Perform stratified k-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Collect predictions from each fold for aggregate evaluation
            all_true = []
            all_pred = []
            all_prob = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)
                
                all_true.extend(y_val)
                all_pred.extend(y_pred)
                all_prob.extend(y_prob)
            
            # Calculate metrics
            metrics = self.evaluate_model(
                np.array(all_true),
                np.array(all_pred),
                np.array(all_prob)
            )
            
            results[name] = metrics
            
            # Update best model based on kappa score
            if metrics['kappa'] > best_kappa:
                best_kappa = metrics['kappa']
                self.best_model = model
                self.best_model_name = name
                
            # Retrain best model on full dataset
            self.best_model.fit(X, y)
        
        return results
    
    def explain_predictions(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Generate SHAP explanations for PHQ-9 predictions."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
            
        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(X)
        
        # Get feature importance for each severity level
        feature_importance = {}
        for i, severity in enumerate(self.severity_levels):
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(shap_values[i]).mean(0)
            }).sort_values('importance', ascending=False)
            feature_importance[severity] = importance
        
        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance
        }
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Make predictions with confidence scores and uncertainty estimates."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
            
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)
        
        # Calculate confidence metrics
        confidence_metrics = {
            'prediction_confidence': probabilities.max(axis=1),
            'prediction_entropy': -np.sum(probabilities * np.log2(probabilities + 1e-10), axis=1),
            'second_best_diff': np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]
        }
        
        return predictions, probabilities, confidence_metrics
    
    def save_model(self, path: str):
        """Save the trained model and metadata."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
            
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'severity_levels': self.severity_levels
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """Load a saved model and metadata."""
        model_data = joblib.load(path)
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.severity_levels = model_data['severity_levels']