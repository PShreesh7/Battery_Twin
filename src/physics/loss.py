import torch
import torch.nn as nn
from typing import Dict, Tuple
from src.physics.residual import PhysicsResidualModule

class PINNCompositeLoss(nn.Module):
    """Composite Loss Function for Physics-Informed Neural Network:
    L_total = L_data + lambda_physics * L_physics
    """
    
    def __init__(
        self,
        equation_name: str,
        physics_params: Dict[str, float],
        lambda_physics: float = 0.05,
        gamma_monotonicity: float = 0.5
    ):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.mse_loss = nn.MSELoss()
        self.physics_residual = PhysicsResidualModule(
            equation_name=equation_name,
            params=physics_params,
            gamma_monotonicity=gamma_monotonicity
        )
        
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        cycles: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute composite PINN loss and return individual loss components for tracking."""
        loss_data = self.mse_loss(y_pred, y_true)
        
        if self.lambda_physics > 0:
            loss_phys, loss_res, loss_mono = self.physics_residual(cycles, y_pred)
            loss_total = loss_data + self.lambda_physics * loss_phys
        else:
            loss_phys = torch.tensor(0.0, device=y_pred.device)
            loss_res = torch.tensor(0.0, device=y_pred.device)
            loss_mono = torch.tensor(0.0, device=y_pred.device)
            loss_total = loss_data
            
        return loss_total, loss_data, loss_phys, loss_mono
