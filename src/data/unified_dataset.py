from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.features.extractor import FeatureExtractor
from src.preprocessing.scaler import FeatureScaler

class BatterySequenceDataset(Dataset):
    """PyTorch Sequence Dataset for sliding window battery prognostics."""
    
    def __init__(
        self,
        X_seq: np.ndarray,
        y_target: np.ndarray,
        cycles: np.ndarray,
        initial_capacities: np.ndarray
    ):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y_target = torch.tensor(y_target, dtype=torch.float32).unsqueeze(-1)
        self.cycles = torch.tensor(cycles, dtype=torch.float32).unsqueeze(-1)
        self.initial_capacities = torch.tensor(initial_capacities, dtype=torch.float32).unsqueeze(-1)
        
    def __len__(self) -> int:
        return len(self.y_target)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.X_seq[idx],
            self.y_target[idx],
            self.cycles[idx],
            self.initial_capacities[idx]
        )

def create_sequences_from_cell(
    cell_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "capacity",
    seq_length: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate sliding window sequences from a single battery cell DataFrame."""
    features = cell_df[feature_cols].values
    targets = cell_df[target_col].values
    cycles = cell_df["cycle"].values
    init_cap = float(cell_df["capacity"].iloc[0])
    
    X_list, y_list, cycle_list, init_cap_list = [], [], [], []
    
    for i in range(len(cell_df) - seq_length + 1):
        X_seq = features[i : i + seq_length]
        y_val = targets[i + seq_length - 1]
        c_val = cycles[i + seq_length - 1]
        
        X_list.append(X_seq)
        y_list.append(y_val)
        cycle_list.append(c_val)
        init_cap_list.append(init_cap)
        
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(cycle_list, dtype=np.float32),
        np.array(init_cap_list, dtype=np.float32)
    )

def prepare_temporal_splits(
    cell_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "capacity",
    seq_length: int = 5,
    train_frac: float = 0.60,
    val_frac: float = 0.15,
    low_data_fraction: float = 1.0
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FeatureScaler]]:
    """Split a single cell chronologically and fit FeatureScaler on train only."""
    n_total = len(cell_df)
    n_train_total = int(n_total * train_frac)
    n_train = int(n_train_total * low_data_fraction)
    n_val = int(n_total * val_frac)
    
    train_df = cell_df.iloc[:n_train].copy()
    val_df = cell_df.iloc[n_train_total : n_train_total + n_val].copy()
    test_df = cell_df.iloc[n_train_total + n_val :].copy()
    
    # Fit scaler strictly on training slice
    scaler = FeatureScaler(feature_cols=feature_cols)
    scaler.fit(train_df)
    
    train_df_scaled = scaler.transform(train_df)
    val_df_scaled = scaler.transform(val_df)
    test_df_scaled = scaler.transform(test_df)
    
    train_data = create_sequences_from_cell(train_df_scaled, feature_cols, target_col, seq_length)
    val_data = create_sequences_from_cell(val_df_scaled, feature_cols, target_col, seq_length)
    test_data = create_sequences_from_cell(test_df_scaled, feature_cols, target_col, seq_length)
    
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "scaler": scaler,
        "raw_splits": (train_df, val_df, test_df)
    }
