import pandas as pd
import numpy as np
from typing import Dict
from src.data_loader import load_clinical_data

def calculate_visit_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate time gaps between visits"""
    df = df.sort_values(['patient_id', 'visit_month'])
    df['months_since_last_visit'] = df.groupby('patient_id')['visit_month'].diff()
    df['visit_frequency'] = df.groupby('patient_id')['visit_month'].transform(
        lambda x: x.diff().mean()
    )
    return df

def calculate_trajectory_slopes(df: pd.DataFrame, target_cols: list) -> Dict[str, pd.Series]:
    """Calculate linear slopes for target variables over time"""
    slopes = {}
    for col in target_cols:
        slope_name = f'{col}_slope'
        df[slope_name] = df.groupby('patient_id').apply(
            lambda x: np.polyfit(x['visit_month'], x[col], 1)[0]
        ).reset_index(level=0, drop=True)
        slopes[col] = df[slope_name]
    return slopes

def calculate_stability_metrics(df: pd.DataFrame, protein_cols: list) -> pd.DataFrame:
    """Calculate protein stability metrics across visits"""
    for protein in protein_cols:
        # Coefficient of variation
        df[f'{protein}_cv'] = df.groupby('patient_id')[protein].transform(
            lambda x: x.std() / x.mean()
        )
        
        # Maximum fold change
        df[f'{protein}_max_fc'] = df.groupby('patient_id')[protein].transform(
            lambda x: x.max() / x.min()
        )
    return df

def create_all_temporal_features(base_path: Path) -> pd.DataFrame:
    """Generate complete set of temporal features"""
    clinical = load_clinical_data(base_path)
    clinical = calculate_visit_intervals(clinical)
    
    # Example protein columns - should match actual data
    protein_cols = ['O00391', 'P05067']  
    
    # Calculate trajectory features
    target_cols = ['updrs_3', 'updrs_3_adj']
    _ = calculate_trajectory_slopes(clinical, target_cols)
    
    # Calculate protein stability
    clinical = calculate_stability_metrics(clinical, protein_cols)
    
    return clinical