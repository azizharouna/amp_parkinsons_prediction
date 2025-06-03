import lightgbm as lgb
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Union
from pydantic import BaseModel, ValidationError, confloat, conint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClinicalInput(BaseModel):
    """Pydantic model for input validation with clinical constraints"""
    visit_month: conint(ge=0, le=120)  # 0-10 years range
    visit_gap: conint(ge=0, le=24)      # Max 2 years between visits
    months_since_first: conint(ge=0, le=120)
    prot_O00391: confloat(ge=0)
    prot_P05067: confloat(ge=0)
    prot_Q9Y6K9: confloat(ge=0)
    prot_O00391_delta: float
    prot_P05067_delta: float
    prot_Q9Y6K9_delta: float
    disease_stage: conint(ge=0, le=3)   # 0=early, 1=mild, 2=moderate, 3=severe
    med_response: confloat(ge=0, le=100) # Percentage scale
    on_medication: conint(ge=0, le=1)   # Binary flag

class ParkinsonPredictor:
    def __init__(self, model_path='modeling/models/updrs_3_adj/model.txt'):
        """Initialize predictor with clinical-grade validation"""
        try:
            self.model = lgb.Booster(model_file=model_path)
            self.model_version = self._extract_model_version(model_path)
            logger.info(f"Loaded model v{self.model_version} from {model_path}")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError("Clinical predictor initialization failed") from e
        
        # Define clinical feature expectations
        self.expected_features = [
            'visit_month', 'visit_gap', 'months_since_first',
            'prot_O00391', 'prot_P05067', 'prot_Q9Y6K9',
            'prot_O00391_delta', 'prot_P05067_delta', 'prot_Q9Y6K9_delta',
            'disease_stage', 'med_response', 'on_medication'
        ]
        
        # Set clinical thresholds
        self.clinical_config = {
            'high_risk_threshold': 35.0,
            'medication_alert_threshold': 0.5  # >50% response needed
        }
    
    def _extract_model_version(self, path: str) -> str:
        """Extract model version from metadata if available"""
        try:
            # LightGBM models can store metadata
            return self.model.params.get('version', '1.0.0')
        except AttributeError:
            return "1.0.0"
    
    def validate_input(self, input_data: pd.DataFrame) -> None:
        """Comprehensive clinical data validation"""
        # 1. Feature existence check
        missing_features = set(self.expected_features) - set(input_data.columns)
        if missing_features:
            logger.error(f"Missing clinical features: {missing_features}")
            raise ValueError(f"Required features missing: {', '.join(missing_features)}")
        
        # 2. Pydantic validation for each row
        errors = []
        for i, row in input_data.iterrows():
            try:
                ClinicalInput(**row.to_dict())
            except ValidationError as e:
                errors.append(f"Row {i}: {str(e)}")
        
        if errors:
            logger.error(f"Input validation failed: {errors}")
            raise ValueError("Clinical data validation failed", errors)
    
    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make UPDRS3 predictions with clinical interpretation
        
        Args:
            input_data: DataFrame with clinical features
            
        Returns:
            DataFrame with predictions and clinical insights
        """
        try:
            # Validate before prediction
            self.validate_input(input_data)
            
            # Ensure feature order
            input_data = input_data[self.expected_features]
            
            # Generate predictions
            predictions = self.model.predict(input_data)
            
            # Create clinical interpretation
            output = input_data.copy()
            output['predicted_updrs3_adj'] = predictions
            output = self.add_clinical_insights(output)
            
            logger.info(f"Predicted {len(output)} clinical samples")
            return output
        except Exception as e:
            logger.exception("Prediction failed")
            raise RuntimeError("Clinical prediction error") from e
    
    def add_clinical_insights(self, results: pd.DataFrame) -> pd.DataFrame:
        """Enhance predictions with clinical interpretation"""
        # 1. Risk stratification
        results['risk_category'] = results['predicted_updrs3_adj'].apply(
            lambda x: 'High' if x > self.clinical_config['high_risk_threshold'] else 'Moderate'
        )
        
        # 2. Medication effectiveness
        results['med_effectiveness'] = results.apply(
            lambda row: 'Inadequate' if (
                row['on_medication'] == 1 and 
                row['med_response'] < self.clinical_config['medication_alert_threshold']
            ) else 'Adequate',
            axis=1
        )
        
        # 3. Biomarker flags
        for prot in ['O00391', 'P05067', 'Q9Y6K9']:
            results[f'{prot}_alert'] = results[f'prot_{prot}_delta'].apply(
                lambda x: 'Critical' if x < -1000 else 'Monitor' if x < 0 else 'Normal'
            )
        
        return results
    
    @staticmethod
    def create_sample_input(num_samples=1) -> pd.DataFrame:
        """Generate clinically valid sample data"""
        return pd.DataFrame({
            'visit_month': np.random.randint(0, 36, num_samples),
            'visit_gap': np.random.randint(1, 6, num_samples),
            'months_since_first': np.random.randint(0, 36, num_samples),
            'prot_O00391': np.random.uniform(10000, 20000, num_samples),
            'prot_P05067': np.random.uniform(400000, 600000, num_samples),
            'prot_Q9Y6K9': np.random.uniform(5000, 10000, num_samples),
            'prot_O00391_delta': np.random.uniform(-5000, 5000, num_samples),
            'prot_P05067_delta': np.random.uniform(-200000, 200000, num_samples),
            'prot_Q9Y6K9_delta': np.random.uniform(-3000, 3000, num_samples),
            'disease_stage': np.random.randint(0, 4, num_samples),
            'med_response': np.random.uniform(0.0, 1.0, num_samples),
            'on_medication': np.random.randint(0, 2, num_samples)
        })
    
    def to_clinical_json(self, results: pd.DataFrame) -> Dict:
        """Convert results to clinician-friendly JSON format"""
        return {
            "model_version": self.model_version,
            "predictions": results.to_dict(orient='records'),
            "clinical_summary": {
                "high_risk_patients": results[results['risk_category'] == 'High'].shape[0],
                "medication_issues": results[results['med_effectiveness'] == 'Inadequate'].shape[0]
            }
        }

# API Integration Example
if __name__ == "__main__":
    # Initialize clinical predictor
    predictor = ParkinsonPredictor()
    
    # Create valid sample data
    sample_data = predictor.create_sample_input(3)
    
    # Generate predictions
    try:
        predictions = predictor.predict(sample_data)
        print("Clinical Predictions:")
        print(predictions[['predicted_updrs3_adj', 'risk_category', 'med_effectiveness']])
        
        # API-ready output
        print("\nJSON Output:")
        print(json.dumps(predictor.to_clinical_json(predictions), indent=2))
    except Exception as e:
        print(f"Clinical error: {str(e)}")