import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

def enable_dropout_at_test(model: nn.Module) -> None:
    """Enable dropout layers during inference for Monte Carlo Dropout."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def predict_mc_dropout(
    model: nn.Module,
    X_tensor: torch.Tensor,
    n_samples: int = 100,
    confidence_level: float = 0.90
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform Monte Carlo Dropout stochastic forward passes.
    Returns: (mean, std, lower_bound, upper_bound)
    """
    model.eval()
    enable_dropout_at_test(model)
    
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X_tensor)
            samples.append(pred.squeeze(-1).cpu().numpy())
            
    samples = np.array(samples)  # Shape: (n_samples, batch_size)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    
    alpha = (1.0 - confidence_level) / 2.0
    lower_bound = np.percentile(samples, alpha * 100, axis=0)
    upper_bound = np.percentile(samples, (1.0 - alpha) * 100, axis=0)
    
    return mean, std, lower_bound, upper_bound
