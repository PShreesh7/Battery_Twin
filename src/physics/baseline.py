import json
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from src.physics.equations import PHYSICS_REGISTRY
from src.physics.fitting import select_best_physics_equation

class PhysicsOnlyBaseline:
    """Standalone Physics-Only Battery Prognostic Model."""
    
    def __init__(self, equation_name: Optional[str] = None):
        self.equation_name = equation_name
        self.params: Dict[str, float] = {}
        self.metrics: Dict[str, float] = {}
        self.is_fitted = False
        
    def fit(self, cycles: np.ndarray, capacities: np.ndarray, q0_initial: float) -> "PhysicsOnlyBaseline":
        """Fit physics degradation equation strictly on training data."""
        if self.equation_name:
            from src.physics.fitting import fit_physics_equation
            self.params, self.metrics = fit_physics_equation(self.equation_name, cycles, capacities, q0_initial)
        else:
            self.equation_name, self.params, self.metrics = select_best_physics_equation(cycles, capacities, q0_initial)
        self.is_fitted = True
        return self
        
    def predict(self, cycles: Union[np.ndarray, list]) -> np.ndarray:
        """Predict capacity for given cycle indices."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted! Call fit() first.")
        model = PHYSICS_REGISTRY[self.equation_name]
        t = np.array(cycles, dtype=np.float64)
        return model(t, *[self.params[k] for k in ["q0", "alpha", "beta"]])
        
    def forecast(self, start_cycle: int, horizon: int) -> np.ndarray:
        """Extrapolate degradation trajectory for future cycles."""
        future_cycles = np.arange(start_cycle, start_cycle + horizon)
        return self.predict(future_cycles)

    def save(self, filepath: str) -> None:
        """Serialize physics model parameters to JSON."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "equation_name": self.equation_name,
            "params": self.params,
            "metrics": self.metrics,
            "is_fitted": self.is_fitted
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "PhysicsOnlyBaseline":
        """Load physics model from JSON artifact."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls(equation_name=data["equation_name"])
        obj.params = data["params"]
        obj.metrics = data["metrics"]
        obj.is_fitted = data["is_fitted"]
        return obj
