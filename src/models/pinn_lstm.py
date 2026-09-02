import torch
import torch.nn as nn
from typing import Dict, Optional
from src.models.base_model import BaseBatteryModel
from src.physics.loss import PINNCompositeLoss

class PINNBatteryLSTM(BaseBatteryModel):
    """Physics-Informed Neural Network (PINN) LSTM for Battery Degradation."""
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        equation_name: str = "power_law",
        physics_params: Optional[Dict[str, float]] = None,
        lambda_physics: float = 0.05,
        gamma_monotonicity: float = 0.5
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.equation_name = equation_name
        self.physics_params = physics_params or {"q0": 2.0, "alpha": 0.005, "beta": 0.75}
        self.lambda_physics = lambda_physics
        self.gamma_monotonicity = gamma_monotonicity
        
        # Sequence Encoder identical to AI-LSTM for fair comparison
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(32, 1)
        
        # PINN Physics Loss Module
        self.criterion = PINNCompositeLoss(
            equation_name=self.equation_name,
            physics_params=self.physics_params,
            lambda_physics=self.lambda_physics,
            gamma_monotonicity=self.gamma_monotonicity
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.dropout(last_step)
        out = self.relu(self.fc1(out))
        out = self.fc_out(out)
        return out
        
    def compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        cycles: torch.Tensor
    ):
        return self.criterion(y_pred, y_true, cycles)
