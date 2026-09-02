import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

class FeatureScaler:
    """Z-score StandardScaler fitted strictly on training data to prevent leakage."""
    
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        """Compute mean and standard deviation on training dataframe only."""
        for col in self.feature_cols:
            if col in df.columns:
                m = float(df[col].mean())
                s = float(df[col].std())
                self.means[col] = m
                self.stds[col] = s if s > 1e-8 else 1.0
        self.is_fitted = True
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling transformation."""
        if not self.is_fitted:
            raise ValueError("FeatureScaler is not fitted! Fit on training data first.")
        df_scaled = df.copy()
        for col in self.feature_cols:
            if col in df_scaled.columns and col in self.means:
                df_scaled[col] = (df_scaled[col] - self.means[col]) / self.stds[col]
        return df_scaled
        
    def inverse_transform_target(self, y_scaled: np.ndarray, target_col: str = "capacity") -> np.ndarray:
        """Inverse scale target variable."""
        if target_col in self.means and target_col in self.stds:
            return y_scaled * self.stds[target_col] + self.means[target_col]
        return y_scaled

    def save(self, filepath: str) -> None:
        """Serialize scaler parameters to JSON."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "feature_cols": self.feature_cols,
            "means": self.means,
            "stds": self.stds,
            "is_fitted": self.is_fitted
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "FeatureScaler":
        """Load scaler parameters from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        scaler = cls(feature_cols=data["feature_cols"])
        scaler.means = data["means"]
        scaler.stds = data["stds"]
        scaler.is_fitted = data["is_fitted"]
        return scaler
