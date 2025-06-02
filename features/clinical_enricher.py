import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import load_clinical_data, load_proteins
from config import PROCESSED_DIR, TARGETS

def enrich_features(base_path: str) -> None:
    """Core clinical enrichment pipeline"""
    base_path = Path(base_path)
    
    # Load datasets
    clinical = load_clinical_data(base_path)
    proteins = load_proteins(base_path)
    
    # Step 1: Medication-adjusted targets
    clinical = _adjust_medication_effect(clinical)
    
    # Step 2: Temporal feature engineering
    clinical = _add_temporal_features(clinical)
    
    # Step 3: Protein merge with biomarker focus
    enriched = _merge_protein_features(clinical, proteins)
    
    # Step 4: Save processed data
    enriched.to_parquet(base_path / PROCESSED_DIR / "enriched_clinical.parquet")
    print(f"✅ Enriched data saved: {base_path / PROCESSED_DIR}")

def _adjust_medication_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Create medication-adjusted UPDRS3 target"""
    med_off_median = df[df['on_medication'] == 0]['updrs_3'].median()
    med_on_median = df[df['on_medication'] == 1]['updrs_3'].median()
    adjustment_factor = med_off_median - med_on_median
    
    df['updrs_3_adj'] = df['updrs_3'] + (adjustment_factor * df['on_medication'])
    return df

def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer time-aware features"""
    df = df.sort_values(['patient_id', 'visit_month'])
    
    # Rate of change features
    for target in TARGETS:
        df[f'{target}_delta'] = df.groupby('patient_id')[target].diff() / \
                                df.groupby('patient_id')['visit_month'].diff()
    
    # Visit gap features
    df['visit_gap'] = df.groupby('patient_id')['visit_month'].diff().fillna(0)
    df['months_since_first'] = df['visit_month'] - df.groupby('patient_id')['visit_month'].transform('min')
    
    # Progression stage classification
    conditions = [
        df['updrs_3'] < 20,
        (df['updrs_3'] >= 20) & (df['updrs_3'] < 40),
        df['updrs_3'] >= 40
    ]
    choices = ['early', 'moderate', 'advanced']
    df['disease_stage'] = np.select(conditions, choices, default='unknown')

    # Medication response metric
    df['med_response'] = df.groupby('patient_id').apply(
        lambda x: x['updrs_3_adj'].diff() / x['visit_month'].diff()
    ).reset_index(level=0, drop=True)
    
    return df

def _merge_protein_features(clinical: pd.DataFrame, proteins: pd.DataFrame) -> pd.DataFrame:
    """Merge proteins with clinical data, focusing on biomarkers"""
    # Pivot proteins to wide format
    protein_wide = proteins.pivot_table(
        index='visit_id', 
        columns='UniProt', 
        values='NPX',
        aggfunc='mean'
    ).add_prefix('prot_')
    
    # Focus on top biomarkers
    TOP_BIOMARKERS = ['O00391', 'P05067', 'Q9Y6K9']
    biomarker_cols = [f'prot_{p}' for p in TOP_BIOMARKERS]
    
    # Handle missing biomarker columns
    for prot in TOP_BIOMARKERS:
        col = f'prot_{prot}'
        if col not in protein_wide.columns:
            protein_wide[col] = np.nan
    
    # Merge with clinical data
    merged = clinical.merge(
        protein_wide[biomarker_cols], 
        on='visit_id', 
        how='left'
    )
    
    # Add biomarker change features
    for prot in TOP_BIOMARKERS:
        merged[f'prot_{prot}_delta'] = merged.groupby('patient_id')[f'prot_{prot}'].diff()
    
    return merged