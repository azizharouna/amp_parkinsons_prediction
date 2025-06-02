import pandas as pd
import numpy as np
from typing import Dict
from src.data_loader import load_clinical_data

class ClinicalDataEnricher:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.medication_adjustments = {}
        
    def calculate_medication_adjustments(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate medication effect adjustments for all UPDRS targets"""
        adjustments = {}
        for target in ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']:
            med_off = df[df['on_medication'] == 0][target].median()
            med_on = df[df['on_medication'] == 1][target].median()
            adjustments[target] = med_off - med_on
        self.medication_adjustments = adjustments
        return adjustments

    def create_adjusted_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create medication-adjusted versions of all targets"""
        for target, adjustment in self.medication_adjustments.items():
            df[f'{target}_adj'] = df[target] + (adjustment * df['on_medication'])
        return df

    def add_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance clinical data with derived features"""
        # Disease progression markers
        df['total_updrs'] = df[['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']].sum(axis=1)
        df['motor_score'] = df['updrs_3'] + df['updrs_4']
        
        # Medication response features
        df['med_response_ratio'] = df['updrs_3'] / df['updrs_3_adj']
        
        # Clinical stage indicators
        df['stage_early'] = (df['total_updrs'] < 30).astype(int)
        df['stage_late'] = (df['total_updrs'] >= 60).astype(int)
        
        return df

    def process(self) -> pd.DataFrame:
        """Run complete clinical data enrichment pipeline"""
        clinical = load_clinical_data(self.base_path)
        self.calculate_medication_adjustments(clinical)
        clinical = self.create_adjusted_targets(clinical)
        clinical = self.add_clinical_features(clinical)
        return clinical