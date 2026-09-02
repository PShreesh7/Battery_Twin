import torch
import torch.nn as nn
from src.models.base_model import BaseBatteryModel

class AIBatteryLSTM(BaseBatteryModel):
    """Purely Data-Driven LSTM Prognostic Architecture (AI Baseline)."""
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch_size, seq_len, input_size) -> (batch_size, 1)"""
        lstm_out, _ = self.lstm(x)
        # Take representation from last time step
        last_step = lstm_out[:, -1, :]
        out = self.dropout(last_step)
        out = self.relu(self.fc1(out))
        out = self.fc_out(out)
        return out
