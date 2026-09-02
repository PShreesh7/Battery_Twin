import pytest
from src.inference.pipeline import BatteryTwinInferencePipeline

def test_inference_pipeline():
    pipeline = BatteryTwinInferencePipeline()
    cap = pipeline.predict_capacity("B0005", model_type="pinn", cycle=100)
    soh = pipeline.predict_soh("B0005", model_type="pinn", cycle=100)
    assert 0.5 <= cap <= 2.5
    assert 0.4 <= soh <= 1.2

def test_forecast_pipeline():
    pipeline = BatteryTwinInferencePipeline()
    fc = pipeline.forecast("B0005", model_type="pinn", start_cycle=100, horizon=15)
    assert len(fc["predicted_capacity"]) == 15
    assert "plausibility" in fc
