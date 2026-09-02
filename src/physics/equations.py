from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union
import numpy as np
import torch

class BasePhysicsEquation(ABC):
    """Abstract Base Class for Battery Degradation Physics Equations."""
    
    @abstractmethod
    def __call__(self, t: Union[np.ndarray, torch.Tensor], *params) -> Union[np.ndarray, torch.Tensor]:
        pass

class PowerLawModel(BasePhysicsEquation):
    """SEI Diffusion-limited degradation: Q(t) = Q0 - alpha * t^beta"""
    name = "power_law"
    
    def __call__(self, t: Union[np.ndarray, torch.Tensor], q0: float, alpha: float, beta: float) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(t, torch.Tensor):
            return q0 - alpha * torch.pow(torch.clamp(t, min=1.0), beta)
        return q0 - alpha * np.power(np.maximum(t, 1.0), beta)

class ExponentialModel(BasePhysicsEquation):
    """Exponential degradation: Q(t) = (Q0 - beta) * exp(-alpha * t) + beta"""
    name = "exponential"
    
    def __call__(self, t: Union[np.ndarray, torch.Tensor], q0: float, alpha: float, beta: float) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(t, torch.Tensor):
            return (q0 - beta) * torch.exp(-alpha * t) + beta
        return (q0 - beta) * np.exp(-alpha * t) + beta

class ModifiedLogisticModel(BasePhysicsEquation):
    """Modified Logistic / Verhulst: Q(t) = Q0 / (1 + alpha * exp(beta * t))"""
    name = "logistic"
    
    def __call__(self, t: Union[np.ndarray, torch.Tensor], q0: float, alpha: float, beta: float) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(t, torch.Tensor):
            return q0 / (1.0 + alpha * torch.exp(torch.clamp(beta * t, max=20.0)))
        return q0 / (1.0 + alpha * np.exp(np.minimum(beta * t, 20.0)))

PHYSICS_REGISTRY = {
    "power_law": PowerLawModel(),
    "exponential": ExponentialModel(),
    "logistic": ModifiedLogisticModel()
}
