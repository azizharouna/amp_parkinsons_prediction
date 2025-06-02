import pandas as pd
from pathlib import Path
from typing import Tuple
from src.data_loader import load_proteins, load_peptides

def aggregate_peptides_to_proteins(base_path: Path) -> pd.DataFrame:
    """Aggregate peptide abundances to protein-level measurements"""
    peptides = load_peptides(base_path)
    proteins = load_proteins(base_path)
    
    # Calculate peptide-protein mapping weights
    peptide_counts = peptides.groupby(['UniProt', 'Peptide']).size().reset_index(name='count')
    total_peptides = peptide_counts.groupby('UniProt')['count'].sum().reset_index(name='total')
    weights = peptide_counts.merge(total_peptides, on='UniProt')
    weights['weight'] = weights['count'] / weights['total']
    
    # Merge with peptide abundances
    weighted_peptides = peptides.merge(
        weights[['UniProt', 'Peptide', 'weight']],
        on=['UniProt', 'Peptide']
    )
    weighted_peptides['weighted_abundance'] = (
        weighted_peptides['PeptideAbundance'] * weighted_peptides['weight']
    )
    
    # Aggregate to protein level
    protein_abundances = weighted_peptides.groupby(
        ['visit_id', 'UniProt']
    )['weighted_abundance'].sum().reset_index()
    
    return protein_abundances

def create_protein_features(base_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate protein-level features from raw data"""
    proteins = load_proteins(base_path)
    peptide_aggregates = aggregate_peptides_to_proteins(base_path)
    
    # Combine with existing protein measurements
    combined = proteins.merge(
        peptide_aggregates,
        on=['visit_id', 'UniProt'],
        how='left',
        suffixes=('_npx', '_peptide')
    )
    
    # Calculate agreement metrics
    combined['measurement_ratio'] = (
        combined['NPX'] / combined['weighted_abundance']
    )
    combined['measurement_diff'] = (
        combined['NPX'] - combined['weighted_abundance']
    )
    
    return combined, proteins