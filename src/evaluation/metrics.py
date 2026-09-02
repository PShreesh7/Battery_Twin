import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression performance metrics."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    
    # Avoid division by zero in MAPE
    denom = np.where(np.abs(y_t) < 1e-6, 1e-6, np.abs(y_t))
    mape = float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)
    
    # R2 score
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
    
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2
    }
