from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseBatteryModel(nn.Module, ABC):
    """Abstract Base Class for Battery Prognostic Neural Network Architectures."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
