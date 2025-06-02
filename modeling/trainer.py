import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
import yaml
import os
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("modeling/models")
TARGETS = ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_3_adj', 'updrs_4']

def load_params(target: str) -> dict:
    """Load target-specific parameters from config"""
    with open("modeling/configs/lgbm_params.yaml") as f:
        params = yaml.safe_load(f)
    return {**params['params'], **params.get(target, {})}

def feature_engineer(df: pd.DataFrame, target: str) -> tuple:
    """Target-specific feature engineering"""
    base_features = [
        'visit_month', 'visit_gap', 'months_since_first',
        'prot_O00391', 'prot_P05067', 'prot_Q9Y6K9',
        'prot_O00391_delta', 'prot_P05067_delta', 'prot_Q9Y6K9_delta'
    ]
    
    if target == 'updrs_3_adj':
        features = base_features + ['disease_stage', 'med_response', 'on_medication']
    elif target == 'updrs_4':
        features = base_features + ['updrs_1_delta', 'updrs_2_delta']
    else:
        features = base_features
    
    return df[features], df[target]

def plot_feature_importance(model, features, target):
    """Save feature importance plot"""
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance['feature'][:15], importance['importance'][:15])
    plt.title(f'{target} Feature Importance')
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/{target}/feature_importance.png")
    plt.close()

def train_progression_models():
    """End-to-end training for all UPDRS targets"""
    df = pd.read_parquet(PROCESSED_DIR / "enriched_clinical.parquet")
    
    # Convert categorical columns to numeric codes
    categorical_cols = ['disease_stage', 'med_response', 'on_medication']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes
    
    for target in TARGETS:
        os.makedirs(f"{MODEL_DIR}/{target}", exist_ok=True)
        X, y = feature_engineer(df, target)
        tscv = TimeSeriesSplit(n_splits=5)
        models = []
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**load_params(target))
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(stopping_rounds=50)]
            )
            models.append(model)
            preds = model.predict(X_val)
            score = 100 * np.mean(np.abs(preds - y_val) / y_val.mean())
            scores.append(score)
            print(f"Fold {fold} {target} MAE%: {score:.2f}")
        
        best_model = models[np.argmin(scores)]
        best_model.booster_.save_model(f"{MODEL_DIR}/{target}/model.txt")
        plot_feature_importance(best_model, X.columns, target)
        print(f"✅ Best {target} model saved (MAE%: {min(scores):.2f})")

if __name__ == "__main__":
    train_progression_models()