from typing import Dict
import numpy as np

def evaluate_uncertainty_calibration(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    nominal_confidence: float = 0.90
) -> Dict[str, float]:
    """Compute Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW)."""
    y = np.asarray(y_true, dtype=np.float64)
    low = np.asarray(lower_bound, dtype=np.float64)
    high = np.asarray(upper_bound, dtype=np.float64)
    
    # Coverage: Fraction of true observations inside [lower, upper]
    inside = (y >= low) & (y <= high)
    picp = float(np.mean(inside) * 100.0)
    
    # Width
    mpiw = float(np.mean(high - low))
    
    # Coverage error
    coverage_error = float(picp - nominal_confidence * 100.0)
    
    return {
        "nominal_confidence": nominal_confidence * 100.0,
        "picp_empirical_coverage": round(picp, 2),
        "mpiw_mean_width": round(mpiw, 4),
        "coverage_error": round(coverage_error, 2),
        "is_well_calibrated": bool(abs(coverage_error) <= 10.0)
    }
