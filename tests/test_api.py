import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "online"

def test_api_models():
    res = client.get("/models")
    assert res.status_code == 200
    assert len(res.json()["available_models"]) == 3

def test_api_battery_info():
    res = client.get("/battery/B0005")
    assert res.status_code == 200
    assert res.json()["battery_id"] == "B0005"

def test_api_predict_capacity():
    res = client.post("/predict/capacity", json={"battery_id": "B0005", "model_type": "pinn", "cycle": 100})
    assert res.status_code == 200
    assert "predicted_capacity" in res.json()

def test_api_forecast():
    res = client.post("/forecast", json={"battery_id": "B0005", "model_type": "pinn", "start_cycle": 100, "horizon": 10})
    assert res.status_code == 200
    assert len(res.json()["predicted_capacity"]) == 10

def test_api_physics_check():
    res = client.post("/physics-check", json={"battery_id": "B0005", "predictions": [1.8, 1.75, 1.70, 1.65]})
    assert res.status_code == 200
    assert res.json()["status"] == "PASS"
