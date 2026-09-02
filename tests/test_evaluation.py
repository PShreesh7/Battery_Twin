import pytest
import numpy as np
from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.physics_checks import PhysicalPlausibilityEngine
from src.uncertainty.calibration import evaluate_uncertainty_calibration

def test_metrics_computation():
    y_true = np.array([1.8, 1.7, 1.6, 1.5])
    y_pred = np.array([1.81, 1.69, 1.61, 1.49])
    m = compute_regression_metrics(y_true, y_pred)
    assert "rmse" in m and "mae" in m and "r2" in m
    assert m["rmse"] < 0.05
    assert m["r2"] > 0.90

def test_physics_plausibility_pass():
    preds = np.array([1.85, 1.80, 1.75, 1.70, 1.65])
    engine = PhysicalPlausibilityEngine()
    res = engine.evaluate_plausibility(preds, "B0005")
    assert res["status"] == "PASS"
    assert res["monotonicity_violations"] == 0

def test_physics_plausibility_fail_jump():
    preds = np.array([1.85, 1.80, 2.50, 1.70])  # Non-physical jump
    engine = PhysicalPlausibilityEngine()
    res = engine.evaluate_plausibility(preds, "B0005")
    assert res["status"] in ["WARNING", "FAIL"]

def test_uncertainty_calibration():
    y_true = np.array([1.5, 1.6, 1.7])
    low = np.array([1.4, 1.5, 1.6])
    high = np.array([1.6, 1.7, 1.8])
    calib = evaluate_uncertainty_calibration(y_true, low, high, nominal_confidence=0.90)
    assert calib["picp_empirical_coverage"] == 100.0
