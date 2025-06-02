import pandas as pd
import numpy as np
from pathlib import Path

def load_clinical_data(base_path: Path) -> pd.DataFrame:
    """Load clinical data with dtype optimization and medication flag handling"""
    dtypes = {
        'visit_id': 'category',
        'patient_id': 'category',
        'visit_month': 'int8',
        'updrs_1': 'float32',
        'updrs_2': 'float32',
        'updrs_3': 'float32',
        'updrs_4': 'float32',
        'upd23b_clinical_state_on_medication': 'category'
    }
    clinical_path = base_path / "data" / "raw" / "train_clinical_data.csv"
    if not clinical_path.exists():
        raise FileNotFoundError(f"Clinical data not found at: {clinical_path}")
    df = pd.read_csv(clinical_path, dtype=dtypes)
    
    # Critical: Convert medication to binary flag
    df['on_medication'] = df['upd23b_clinical_state_on_medication'].eq('On').astype('int8')
    
    # Calculate medication adjustment factor
    med_off_median = df[df['upd23b_clinical_state_on_medication']=='Off']['updrs_3'].median()
    med_on_median = df[df['upd23b_clinical_state_on_medication']=='On']['updrs_3'].median()
    adjustment = med_off_median - med_on_median
    
    # Create adjusted target
    df['updrs_3_adj'] = df['updrs_3'] + adjustment * df['on_medication']
    
    return df

def load_peptides(base_path: Path) -> pd.DataFrame:
    """Peptide data with aggressive downcasting"""
    dtypes = {
        'visit_id': 'category',
        'patient_id': 'category',
        'visit_month': 'int8',
        'UniProt': 'category',
        'Peptide': 'category',
        'PeptideAbundance': 'float32'
    }
    return pd.read_csv(base_path / "data\\raw\\train_peptides.csv", dtype=dtypes)

def load_proteins(base_path: Path) -> pd.DataFrame:
    """Protein data (pre-aggregated)"""
    dtypes = {
        'visit_id': 'category',
        'patient_id': 'category',
        'visit_month': 'int8',
        'UniProt': 'category',
        'NPX': 'float32'
    }
    return pd.read_csv(base_path / "data/raw/train_proteins.csv", dtype=dtypes)