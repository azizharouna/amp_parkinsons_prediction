from pathlib import Path
import pandas as pd
from .clinical_enricher import ClinicalDataEnricher
from .protein_processor import create_protein_features
from .temporal_features import create_all_temporal_features
from src.data_loader import load_clinical_data, load_peptides, load_proteins

class FeaturePipeline:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.artifacts = {}
        
    def run_clinical_pipeline(self) -> pd.DataFrame:
        """Run complete clinical data processing"""
        enricher = ClinicalDataEnricher(self.base_path)
        clinical = enricher.process()
        clinical = create_all_temporal_features(self.base_path)
        self.artifacts['clinical'] = clinical
        return clinical
        
    def run_protein_pipeline(self) -> pd.DataFrame:
        """Process protein and peptide data"""
        proteins, _ = create_protein_features(self.base_path)
        self.artifacts['proteins'] = proteins
        return proteins
        
    def merge_features(self) -> pd.DataFrame:
        """Combine all feature sets"""
        clinical = self.artifacts.get('clinical', self.run_clinical_pipeline())
        proteins = self.artifacts.get('proteins', self.run_protein_pipeline())
        
        # Merge on visit_id
        features = clinical.merge(
            proteins.groupby('visit_id').mean(),
            on='visit_id',
            how='left'
        )
        
        # Add patient-level aggregates
        patient_features = proteins.groupby('patient_id').agg(['mean', 'std'])
        patient_features.columns = ['_'.join(col) for col in patient_features.columns]
        features = features.merge(
            patient_features,
            on='patient_id',
            how='left'
        )
        
        self.artifacts['features'] = features
        return features

    def run(self, save_path: str = None) -> pd.DataFrame:
        """Execute complete pipeline"""
        self.run_clinical_pipeline()
        self.run_protein_pipeline()
        features = self.merge_features()
        
        if save_path:
            features.to_parquet(Path(save_path) / 'processed_features.parquet')
            
        return features