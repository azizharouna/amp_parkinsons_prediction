import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from typing import Dict

MODEL_DIR = Path("modeling/models")

def load_models() -> Dict[str, lgb.Booster]:
    """Load all trained models"""
    models = {}
    targets = ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_3_adj', 'updrs_4']
    
    for target in targets:
        model_path = MODEL_DIR / target / "model.txt"
        if model_path.exists():
            models[target] = lgb.Booster(model_file=str(model_path))
    
    return models

def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare API data for model consumption"""
    # Add required temporal features
    data['visit_gap'] = data.groupby('patient_id')['visit_month'].diff().fillna(0)
    data['months_since_first'] = data['visit_month'] - data.groupby('patient_id')['visit_month'].transform('min')
    return data

def predict_test_set(api_data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Generate predictions for Kaggle API"""
    models = load_models()
    processed_data = preprocess_input(api_data)
    results = {}
    
    for target, model in models.items():
        # Handle medication state for test data
        if target == 'updrs_3_adj':
            processed_data['on_medication'] = 1  # Default assumption
            
        results[target] = model.predict(processed_data)
        
    return results

def generate_submission(api_data: pd.DataFrame) -> pd.DataFrame:
    """Format predictions for Kaggle submission"""
    preds = predict_test_set(api_data)
    submission = pd.DataFrame({
        'visit_id': api_data['visit_id'],
        **{f'updrs_{i}': preds[f'updrs_{i}'] for i in range(1,5)},
        'updrs_3_adj': preds.get('updrs_3_adj', np.zeros(len(api_data)))
    })
    return submission