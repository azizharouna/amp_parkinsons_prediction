import numpy as np

def clinical_smape(y_true, y_pred, target: str) -> float:
    """Competition SMAPE with clinical weighting"""
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    # Clinical severity weighting
    if target == 'updrs_4':
        return smape * 1.3  # Weight complications more heavily
    elif target == 'updrs_3_adj':
        return smape * 1.2  # Motor symptoms have higher clinical impact
    return smape

def medication_effect_error(y_true, y_pred, medication_status) -> float:
    """Specialized error metric for medication response"""
    on_med_mask = medication_status == 1
    off_med_error = np.mean(np.abs(y_pred[~on_med_mask] - y_true[~on_med_mask]))
    on_med_error = np.mean(np.abs(y_pred[on_med_mask] - y_true[on_med_mask]))
    return 0.7 * off_med_error + 0.3 * on_med_error  # Prioritize unmedicated state

def clinical_mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error with clinical clipping"""
    # Clip predictions to avoid extreme values
    y_pred = np.clip(y_pred, 0, 100)
    return 100 * np.mean(np.abs((y_pred - y_true) / np.maximum(1, y_true)))