import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_clinical_data, load_peptides, load_proteins

BASE_PATH = PROJECT_ROOT / "data/raw"

def run_health_checks():
    print("=== LOADING DATA ===")
    clinical = load_clinical_data(BASE_PATH)
    peptides = load_peptides(BASE_PATH)
    proteins = load_proteins(BASE_PATH)
    
    print("\n=== CLINICAL DATA ===")
    print(f"Patients: {clinical['patient_id'].nunique()}")
    print(f"Visits: {len(clinical)}")
    print("Medication States:")
    print(clinical['upd23b_clinical_state_on_medication'].value_counts())
    
    print("\n=== PEPTIDE-PROTEIN RELATIONSHIPS ===")
    peptide_prot_map = peptides.groupby('UniProt')['Peptide'].nunique()
    print("Peptides per protein statistics:")
    print(peptide_prot_map.describe())
    
    print("\n=== MEMORY USAGE (MB) ===")
    for name, df in [('clinical', clinical), ('peptides', peptides), ('proteins', proteins)]:
        print(f"{name}: {df.memory_usage(deep=True).sum() / 1024**2:.2f}")

    # Visualization
    plt.figure(figsize=(8,5))
    clinical.boxplot(column='updrs_3', by='upd23b_clinical_state_on_medication', grid=False)
    plt.title('UPDRS3 Motor Symptoms by Medication State')
    plt.suptitle('')
    plt.savefig(PROJECT_ROOT / 'notebooks/medication_impact.png')
    print("\nSaved medication impact plot")

if __name__ == "__main__":
    run_health_checks()