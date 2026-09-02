import pytest
import torch
import numpy as np
from src.physics.equations import PowerLawModel, ExponentialModel, ModifiedLogisticModel
from src.physics.fitting import fit_physics_equation, select_best_physics_equation
from src.physics.residual import PhysicsResidualModule

def test_physics_equations_numpy():
    t = np.array([1.0, 50.0, 100.0])
    power = PowerLawModel()
    q_pow = power(t, q0=2.0, alpha=0.01, beta=0.8)
    assert len(q_pow) == 3
    assert q_pow[0] > q_pow[-1]

def test_physics_equations_torch_grad():
    t = torch.tensor([1.0, 20.0, 50.0], requires_grad=False)
    pred_q = torch.tensor([1.8, 1.6, 1.4], requires_grad=True)
    residual_mod = PhysicsResidualModule("power_law", {"q0": 1.85, "alpha": 0.01, "beta": 0.8})
    loss_tot, loss_res, loss_mono = residual_mod(t, pred_q)
    loss_tot.backward()
    assert pred_q.grad is not None
    assert torch.all(torch.isfinite(pred_q.grad))

def test_physics_fitting():
    t = np.arange(1, 60)
    q = 1.85 - 0.005 * (t ** 0.8)
    params, metrics = fit_physics_equation("power_law", t, q, q0_initial=1.85)
    assert "alpha" in params and "beta" in params
    assert metrics["r2"] > 0.95
