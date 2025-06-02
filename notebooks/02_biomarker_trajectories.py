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

# Validated biomarkers (Q9Y6K9 removed - no measurements)
TOP_BIOMARKERS = ['O00391', 'P05067']

def plot_biomarker_trajectories(protein_id: str, n_patients=5):
    """Plot protein vs UPDRS3 trajectories for sample patients."""
    prot_data = proteins[proteins['UniProt']==protein_id].merge(
        clinical, on=['visit_id', 'patient_id'])
    
    plt.figure(figsize=(10,4))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    for pid in prot_data['patient_id'].unique()[:n_patients]:
        df = prot_data[prot_data['patient_id']==pid].sort_values('visit_month_x')
        ax1.plot(df['visit_month_x'], df['NPX'], 'o-', label=f'Protein {protein_id}')
        ax2.plot(df['visit_month_x'], df['updrs_3'], 'r--', label='UPDRS3')
    
    ax1.set_xlabel('Visit Month')
    ax1.set_ylabel('NPX (Normalized Protein Expression)')
    ax2.set_ylabel('UPDRS3 Score', color='r')
    plt.title(f'{protein_id} vs UPDRS3 Trajectories')
    plt.tight_layout()
    plt.savefig(f'results/{protein_id}_trajectories.png')
    plt.close()

if __name__ == '__main__':
    for prot in TOP_BIOMARKERS:
        plot_biomarker_trajectories(prot)
        print(f'Saved trajectory plot for {prot}')