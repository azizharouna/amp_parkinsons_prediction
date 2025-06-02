# Project Journal: Parkinson's Biomarker Prediction

## Project Genesis (2025-06-02)
**Purpose**: Develop predictive models for Parkinson's disease progression using protein biomarkers.

### Core Architecture
```
amp_parkinsons_prediction/
├── data/                   # NEVER commit raw data (use .gitignore)
│   ├── raw/                # Original CSVs (train_peptides, etc.)
│   ├── processed/          # Engineered features (parquet format)
│   └── supplemental/       # Additional clinical data
├── features/               # Feature pipelines
├── modeling/               # Training and evaluation
├── notebooks/              # Exploration and analysis
├── api/                    # Kaggle submission tools
└── src/                    # Core utilities
```

## Key Implementation Journey

### 1. Protein Processing Pipeline (2025-06-02)
**Purpose**: Transform raw peptide data into protein-level features

**Technical Decisions**:
- Used weighted aggregation of peptide abundances
- Saved as parquet for efficient storage
- Auto-created processed directory structure

**Challenges Resolved**:
- Fixed Windows path handling issues
- Added pyarrow dependency for parquet support
- Implemented robust directory creation

**Artifacts**:
- `features/protein_processor.py`
- `data/processed/protein_features.parquet`

### 2. Clinical Data Integration (Pending)
**Next Steps**:
- Merge protein features with clinical records
- Adjust for medication effects
- Engineer temporal features

## How To Continue Tomorrow
1. Run clinical enricher:
   ```python
   from features.clinical_enricher import enrich_features
   enrich_features('.')
   ```
2. Verify outputs in `data/processed/`
3. Check `docs/milestone-*.md` for latest progress

## Emergency Context
If completely lost:
1. See `config.py` for all paths and constants
2. Run `python -m pytest tests/` to verify system
3. Check `notebooks/EDA.ipynb` for data understanding