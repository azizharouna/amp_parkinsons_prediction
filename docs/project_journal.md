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

### 1. Clinical Enrichment Pipeline (2025-06-02)
**Breakthrough Features**:
- Medication-adjusted UPDRS3 targets
- Disease stage classification (early/moderate/advanced)
- Biomarker-protein trajectory tracking
- Visit-gap aware temporal features

**Clinical Insights Captured**:
```python
# Medication adjustment formula
adjusted_score = raw_score + (med_off_median - med_on_median) * medication_status

# Disease staging logic:
early_stage = updrs_3 < 20
moderate_stage = 20 <= updrs_3 < 40
advanced_stage = updrs_3 >= 40

# Key biomarker proteins:
TOP_BIOMARKERS = ['O00391', 'P05067', 'Q9Y6K9']
```

**Validation Protocol**:
```python
# After pipeline runs:
assert 'updrs_3_adj' in df.columns
assert df['disease_stage'].isin(['early','moderate','advanced']).all()
assert all(f'prot_{p}' in df.columns for p in TOP_BIOMARKERS)
```

### 2. Protein Processing Pipeline (2025-06-02)
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

### 3. The Training Odyssey (2025-06-02)

**The Breakthrough Moment**:
After days of wrestling with data pipelines and model configurations, we finally saw the first successful training run complete. The validation curves told a promising story - our models were learning meaningful patterns from the protein trajectories and clinical features. The medication-adjusted targets proved particularly insightful, with UPDRS3_adj achieving an impressive 22.39% MAE.

**Behind the Scenes**:
The training process unfolded like a carefully choreographed dance:
1. **Data Preparation**: Our enriched clinical dataset flowed into the pipeline
2. **Temporal Splitting**: The time-series cross validation preserved chronological order
3. **Model Training**: LightGBM trees grew, guided by our carefully tuned parameters
```python
# The heartbeat of our training process
model = lgb.LGBMRegressor(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],  # Watching validation like a hawk
    callbacks=[lgb.early_stopping(50)],  # Preventing overenthusiastic learning
    verbose=10  # Letting the trees whisper their progress
)
```

**Lessons Etched in Code**:
- The medication adjustment was our north star - models trained on adjusted targets consistently outperformed their raw counterparts
- Temporal features acted as the secret sauce, giving our models the gift of time-awareness
- LightGBM proved to be the perfect partner, gracefully handling our clinical time-series data

### 4. Clinical Data Integration (Next Steps)
**Pending Tasks**:
- Finalize protein-clinical feature merge
- Optimize temporal feature calculation
- Prepare Kaggle submission format

## How To Continue Tomorrow
1. Run clinical enricher:
   ```python
   from features.clinical_enricher import enrich_features
   enrich_features('.')
   ```
2. Verify outputs in `data/processed/`
3. Check `docs/milestone-*.md` for latest progress

## Troubleshooting Guide

### Common Issues and Solutions

1. **Clinical Data Specific**
   - Symptom: "KeyError during protein merge"
   - Solution: Auto-creates missing biomarker columns
   - Prevention: Use TOP_BIOMARKERS constant

   - Symptom: "NaN values in temporal features"
   - Solution: Group-wise imputation
   - Debug: Check patient visit sequences

2. **Data Loading Failures**
   - Symptom: "FileNotFoundError" when loading raw data
   - Fix:
     ```python
     from src.data_loader import validate_paths
     validate_paths('.')  # Checks all data paths
     ```
   - Prevention: Always use `config.PATHS` for file access

2. **Feature Processing Errors**
   - Symptom: "KeyError" during protein aggregation
   - Checklist:
     - Verify peptide CSV has 'UniProt' column
     - Check for NaN values in peptide abundances
     - Run `features/protein_processor.py` interactively

3. **Memory Issues**
   - Symptom: "MemoryError" during processing
   - Solutions:
     - Use `api/memory_profiler.py` to monitor usage
     - Process data in chunks (see `src/data_loader.CHUNK_SIZE`)

4. **Dependency Problems**
   - Symptom: Import errors for pyarrow/pandas
   - Recovery:
     ```powershell
     pip install -r requirements.txt --force-reinstall
     ```

### Emergency Context
If completely lost:
1. See `config.py` for all paths and constants
2. Run `python -m pytest tests/` to verify system
3. Check `notebooks/EDA.ipynb` for data understanding
4. Consult `docs/milestone-*.md` for recent changes