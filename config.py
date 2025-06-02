# PATHS (update per environment)
RAW_DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# TARGETS + TIME
TARGETS = [f"updrs_{i}" for i in range(1,5)]
TIME_FEATURES = ["visit_month", "patient_id", "visit_id"]