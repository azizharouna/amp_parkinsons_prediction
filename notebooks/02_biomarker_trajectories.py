import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import load_clinical_data, load_proteins

# Load data
clinical = load_clinical_data(Path('data/raw'))
proteins = load_proteins(Path('data/raw'))

# Top biomarkers from lead-lag analysis
TOP_BIOMARKERS = ['O00391', 'P05067', 'Q9Y6K9']

def plot_biomarker_trajectories(protein_id: str, n_patients=5):
    """Plot protein vs UPDRS3 trajectories for sample patients."""
    prot_data = proteins[proteins['UniProt']==protein_id].merge(
        clinical, on='visit_id')
    
    plt.figure(figsize=(10,4))
    for pid in prot_data['patient_id'].unique()[:n_patients]:
        df = prot_data[prot_data['patient_id']==pid]
        plt.plot(df['visit_month'], df['NPX'], 'o-', label=f'Protein {protein_id}')
        plt.twinx().plot(df['visit_month'], df['updrs_3'], 'r--', label='UPDRS3')
    
    plt.title(f'{protein_id} vs UPDRS3 Trajectories')
    plt.xlabel('Visit Month')
    plt.ylabel('NPX (Normalized Protein Expression)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/{protein_id}_trajectories.png')
    plt.close()

if __name__ == '__main__':
    for prot in TOP_BIOMARKERS:
        plot_biomarker_trajectories(prot)
        print(f'Saved trajectory plot for {prot}')