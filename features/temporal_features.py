import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
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
        if protein in df.columns:
            try:
                # Coefficient of variation
                df[f'{protein}_cv'] = df.groupby('patient_id', observed=True)[protein].transform(
                    lambda x: x.std() / x.mean() if x.mean() != 0 else np.nan
                )
                
                # Maximum fold change (skip if min is 0 to avoid division by zero)
                non_zero = df[protein].replace(0, np.nan)
                df[f'{protein}_max_fc'] = df.groupby('patient_id', observed=True)[protein].transform(
                    lambda x: x.max() / x.min() if x.min() > 0 else np.nan
                )
            except Exception as e:
                print(f"Skipping {protein} due to error: {str(e)}")
    return df

def create_all_temporal_features(base_path: Path) -> pd.DataFrame:
    """Generate complete set of temporal features"""
    clinical = load_clinical_data(base_path)
    clinical = calculate_visit_intervals(clinical)
    
    # Load processed protein features
    protein_features = pd.read_parquet(base_path / "data/processed/protein_features.parquet")
    
    # Merge with clinical data
    clinical = clinical.merge(
        protein_features,
        on=['patient_id', 'visit_month'],
        how='left'
    )
    
    # Get protein columns (UniProt IDs prefixed with NPX_)
    protein_cols = [col for col in protein_features.columns if col.startswith('NPX_')]
    
    # Calculate trajectory features
    target_cols = ['updrs_3', 'updrs_3_adj']
    _ = calculate_trajectory_slopes(clinical, target_cols)
    
    # Calculate protein stability
    clinical = calculate_stability_metrics(clinical, protein_cols)
    
    return clinical