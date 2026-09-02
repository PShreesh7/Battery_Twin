from typing import Any, Dict, Tuple
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.physics.equations import PHYSICS_REGISTRY
from src.utils.logger import setup_logger

logger = setup_logger("physics_fitting")

def fit_physics_equation(
    equation_name: str,
    cycles: np.ndarray,
    capacities: np.ndarray,
    q0_initial: float
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Fit candidate physics degradation model to empirical training data."""
    model = PHYSICS_REGISTRY.get(equation_name)
    if model is None:
        raise ValueError(f"Unknown physics equation: {equation_name}. Choose from {list(PHYSICS_REGISTRY.keys())}")
        
    t = np.array(cycles, dtype=np.float64)
    y = np.array(capacities, dtype=np.float64)
    
    if equation_name == "power_law":
        # Params: (q0, alpha, beta)
        p0 = [q0_initial, 0.005, 0.75]
        bounds = ([q0_initial * 0.95, 1e-6, 0.1], [q0_initial * 1.05, 0.1, 2.0])
        popt, _ = curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=5000)
        params = {"q0": float(popt[0]), "alpha": float(popt[1]), "beta": float(popt[2])}
        
    elif equation_name == "exponential":
        # Params: (q0, alpha, beta)
        p0 = [q0_initial, 0.002, 0.5]
        bounds = ([q0_initial * 0.95, 1e-6, 0.0], [q0_initial * 1.05, 0.05, q0_initial * 0.9])
        popt, _ = curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=5000)
        params = {"q0": float(popt[0]), "alpha": float(popt[1]), "beta": float(popt[2])}
        
    elif equation_name == "logistic":
        # Params: (q0, alpha, beta)
        p0 = [q0_initial, 0.01, 0.005]
        bounds = ([q0_initial * 0.95, 1e-5, 1e-5], [q0_initial * 1.05, 1.0, 0.05])
        popt, _ = curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=5000)
        params = {"q0": float(popt[0]), "alpha": float(popt[1]), "beta": float(popt[2])}
        
    y_pred = model(t, *[params[k] for k in ["q0", "alpha", "beta"]])
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))
    r2 = float(r2_score(y, y_pred))
    
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    logger.info(f"Fitted {equation_name}: params={params}, Train RMSE={rmse:.4f}, R2={r2:.4f}")
    return params, metrics

def select_best_physics_equation(
    cycles: np.ndarray,
    capacities: np.ndarray,
    q0_initial: float
) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """Fit all candidate equations and select the one with highest R2 on training data."""
    best_name = ""
    best_r2 = -float("inf")
    best_params: Dict[str, float] = {}
    best_metrics: Dict[str, float] = {}
    
    for eq_name in PHYSICS_REGISTRY.keys():
        try:
            params, metrics = fit_physics_equation(eq_name, cycles, capacities, q0_initial)
            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                best_name = eq_name
                best_params = params
                best_metrics = metrics
        except Exception as e:
            logger.warning(f"Fitting {eq_name} failed: {e}")
            
    logger.info(f"Selected Best Physics Model: {best_name} with R2={best_r2:.4f}")
    return best_name, best_params, best_metrics
