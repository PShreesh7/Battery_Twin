import pytest
import torch
from src.models.ai_lstm import AIBatteryLSTM
from src.models.pinn_lstm import PINNBatteryLSTM

def test_ai_lstm_forward():
    model = AIBatteryLSTM(input_size=6, hidden_size=32, num_layers=2)
    x = torch.randn(8, 5, 6)
    out = model(x)
    assert out.shape == (8, 1)

def test_pinn_lstm_forward_and_loss():
    model = PINNBatteryLSTM(input_size=6, hidden_size=32, num_layers=2, lambda_physics=0.05)
    x = torch.randn(8, 5, 6)
    y_true = torch.randn(8, 1)
    cycles = torch.tensor([[10.0], [11.0], [12.0], [13.0], [14.0], [15.0], [16.0], [17.0]])
    y_pred = model(x)
    assert y_pred.shape == (8, 1)
    loss_tot, l_data, l_phys, _ = model.compute_loss(y_pred, y_true, cycles)
    assert loss_tot.item() >= 0.0
    loss_tot.backward()
