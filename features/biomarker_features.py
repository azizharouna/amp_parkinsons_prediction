from typing import Dict
import pandas as pd

TOP_BIOMARKERS = ['O00391', 'P05067', 'Q9Y6K9']

def create_biomarker_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create predictive features from top biomarkers.
    
    Args:
        df: Merged clinical and protein DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    # Rate-of-change features
    for prot in TOP_BIOMARKERS:
        df[f'{prot}_delta'] = (
            df.groupby('patient_id')[prot].diff() / 
            df.groupby('patient_id')['visit_month'].diff()
        )
    
    # Medication interaction terms
    if 'on_medication' in df.columns:
        df['O00391_med_interact'] = df['O00391'] * df['on_medication']
    
    # Cumulative exposure
    for prot in TOP_BIOMARKERS:
        df[f'{prot}_cumulative'] = df.groupby('patient_id')[prot].cumsum()
    
    # Clinical impact score
    df['biomarker_impact_score'] = (
        0.42 * df['O00391_delta'].fillna(0) +
        0.38 * df['P05067_delta'].fillna(0) +
        0.35 * df['Q9Y6K9_delta'].fillna(0)
    )
    
    return df

def get_feature_descriptions() -> Dict[str, str]:
    """Return descriptions of all engineered features."""
    return {
        **{f'{prot}_delta': f'Rate of change for protein {prot}' 
           for prot in TOP_BIOMARKERS},
        **{f'{prot}_cumulative': f'Cumulative exposure to protein {prot}'
           for prot in TOP_BIOMARKERS},
        'O00391_med_interact': 'Interaction between O00391 and medication state',
        'biomarker_impact_score': 'Weighted combination of top biomarker deltas'
    }