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
    df = pd.read_csv(base_path / "train_clinical_data.csv", dtype=dtypes)
    
    # Critical: Convert medication to binary flag
    df['on_medication'] = df['upd23b_clinical_state_on_medication'].eq('On').astype('int8')
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
    return pd.read_csv(base_path / "train_peptides.csv", dtype=dtypes)

def load_proteins(base_path: Path) -> pd.DataFrame:
    """Protein data (pre-aggregated)"""
    dtypes = {
        'visit_id': 'category',
        'patient_id': 'category',
        'visit_month': 'int8',
        'UniProt': 'category',
        'NPX': 'float32'
    }
    return pd.read_csv(base_path / "train_proteins.csv", dtype=dtypes)