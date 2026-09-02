from typing import List
import numpy as np
import pandas as pd

class FeatureExtractor:
    """Feature Engineering Pipeline for Battery Prognostics."""
    
    DEFAULT_FEATURE_COLS = [
        "cycle",
        "voltage_mean",
        "current_mean",
        "temp_mean",
        "temp_max",
        "capacity_prev"
    ]
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute lagged capacity, rolling degradation rate, and temperature rise features."""
        df = df.copy()
        
        # Lagged capacity (capacity_prev)
        if "capacity" in df.columns:
            df["capacity_prev"] = df.groupby("cell_id")["capacity"].shift(1)
            # Impute first cycle capacity_prev with initial capacity
            df["capacity_prev"] = df["capacity_prev"].fillna(df["capacity"])
            
            # Capacity delta (rate of degradation)
            df["capacity_delta"] = df["capacity"] - df["capacity_prev"]
            
            # Rolling 5-cycle degradation slope
            df["rolling_decay_rate"] = (
                df.groupby("cell_id")["capacity"]
                .transform(lambda x: x.diff(periods=5) / 5.0)
                .fillna(0.0)
            )
            
        # Temperature delta (heating above ambient)
        if "temp_max" in df.columns and "ambient_temperature" in df.columns:
            df["temp_rise"] = df["temp_max"] - df["ambient_temperature"]
            
        # Voltage spread
        if "voltage_max" in df.columns and "voltage_min" in df.columns:
            df["voltage_range"] = df["voltage_max"] - df["voltage_min"]
            
        return df
