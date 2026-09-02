from typing import Dict, Tuple
import torch
import torch.nn as nn
from src.physics.equations import PHYSICS_REGISTRY

class PhysicsResidualModule(nn.Module):
    """PyTorch Differentiable Residual & Monotonicity Constraint Module."""
    
    def __init__(self, equation_name: str, params: Dict[str, float], gamma_monotonicity: float = 0.5):
        super().__init__()
        self.equation_name = equation_name
        self.params = params
        self.gamma = gamma_monotonicity
        self.model = PHYSICS_REGISTRY[equation_name]
        
    def forward(
        self,
        cycle_tensor: torch.Tensor,
        predicted_capacity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute physics trajectory residual loss and monotonicity violation loss."""
        q_phys = self.model(
            cycle_tensor,
            self.params["q0"],
            self.params["alpha"],
            self.params["beta"]
        )
        
        loss_residual = torch.mean((predicted_capacity - q_phys) ** 2)
        
        if predicted_capacity.shape[0] > 1:
            diff = predicted_capacity[1:] - predicted_capacity[:-1]
            loss_monotonicity = torch.mean(torch.relu(diff) ** 2)
        else:
            loss_monotonicity = torch.tensor(0.0, device=predicted_capacity.device)
            
        total_physics_loss = loss_residual + self.gamma * loss_monotonicity
        return total_physics_loss, loss_residual, loss_monotonicity
