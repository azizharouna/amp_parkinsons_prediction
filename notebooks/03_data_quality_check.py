import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import load_proteins

def analyze_protein_data(protein_id: str):
    """Check data quality for a specific protein"""
    proteins = load_proteins(Path('data/raw'))
    prot_data = proteins[proteins['UniProt'] == protein_id]
    
    print(f"\n=== {protein_id} Data Quality Report ===")
    print(prot_data['NPX'].describe())
    
    # Plot distribution
    plt.figure(figsize=(10,4))
    plt.hist(prot_data['NPX'], bins=50)
    plt.title(f'{protein_id} Distribution')
    plt.xlabel('NPX Value')
    plt.ylabel('Count')
    plt.savefig(f'results/{protein_id}_distribution.png')
    plt.close()
    
    # Check missing values
    print(f"\nMissing Values: {prot_data['NPX'].isna().sum()}/{len(prot_data)}")
    
    # Check constant values
    if prot_data['NPX'].nunique() == 1:
        print("WARNING: Constant values detected!")
    
    return prot_data

if __name__ == '__main__':
    analyze_protein_data('Q9Y6K9')
    analyze_protein_data('O00391')  # For comparison