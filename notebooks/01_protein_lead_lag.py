import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import load_clinical_data, load_proteins
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr

def calculate_lead_lag(clinical_df, protein_df, top_n=20):
    """Calculate protein lead-lag correlations with UPDRS3"""
    # Get top abundant proteins
    top_proteins = protein_df.groupby('UniProt')['NPX'].mean().nlargest(top_n).index
    
    results = []
    protein_df = protein_df.merge(
        clinical_df[['visit_id', 'updrs_3', 'on_medication']],
        on='visit_id'
    )
    
    for prot in tqdm(top_proteins):
        prot_data = protein_df[protein_df['UniProt'] == prot]
        
        # Calculate 1-visit lead
        prot_data[f'{prot}_lead'] = prot_data.groupby(
            ['patient_id', 'on_medication']
        )['NPX'].shift(-1)
        
        # Correlate with NEXT visit's UPDRS3
        valid = prot_data.dropna(subset=[f'{prot}_lead', 'updrs_3'])
        if len(valid) > 10:  # Minimum samples
            corr, pval = pearsonr(valid[f'{prot}_lead'], valid['updrs_3'])
            results.append({
                'protein': prot,
                'correlation': corr,
                'p_value': pval,
                'n_samples': len(valid)
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    base_path = Path("data/raw")
    clinical = load_clinical_data(base_path)
    proteins = load_proteins(base_path)
    
    results = calculate_lead_lag(clinical, proteins)
    results.to_csv("data/processed/protein_lead_lag.csv", index=False)
    print(f"Saved results for {len(results)} proteins")