import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import yaml
import joblib
from pathlib import Path
from src.data_loader import load_processed_data
from src.eval_metrics import clinical_smape

# Load configuration
with open('modeling/configs/lgbm_params.yaml') as f:
    params = yaml.safe_load(f)

def train_model():
    # Load processed data
    df = load_processed_data()
    
    # Feature engineering
    X = df[['O00391', 'P05067', 'O00391_UPDRS3_interact']]
    y = df['updrs_3_adj']
    
    # Time-based cross validation
    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['rmse', 'mae']
        )
        
        # Evaluate
        preds = model.predict(X_val)
        score = clinical_smape(y_val, preds)
        scores.append(score)
        models.append(model)
    
    # Save best model
    best_idx = scores.index(min(scores))
    joblib.dump(models[best_idx], 'modeling/models/best_lgbm.pkl')
    
    print(f"Average Clinical SMAPE: {sum(scores)/len(scores):.2f}")
    return models[best_idx]

if __name__ == '__main__':
    # Create required directories
    Path('modeling/models').mkdir(exist_ok=True)
    train_model()