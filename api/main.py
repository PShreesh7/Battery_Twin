import sys
from pathlib import Path
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictCapacityResponse,
    PredictSOHResponse,
    ForecastRequest,
    ForecastResponse,
    UncertaintyRequest,
    UncertaintyResponse,
    PhysicsCheckRequest,
    PhysicsCheckResponse
)
from src.inference.pipeline import BatteryTwinInferencePipeline
from src.utils.logger import setup_logger

logger = setup_logger("api")

app = FastAPI(
    title="Battery Twin REST API",
    description="Research-grade Physics-Informed Neural Network Digital Twin for Battery Health and Degradation Prognostics",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = BatteryTwinInferencePipeline()

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse()

@app.get("/models", tags=["System"])
def list_models():
    return {
        "available_models": [
            {"id": "physics", "name": "Physics-Only Baseline (Empirical Power Law / Exponential)"},
            {"id": "ai", "name": "AI-Only Baseline (Pure Data-Driven LSTM)"},
            {"id": "pinn", "name": "Hybrid Physics-Informed Neural Network (PINN-LSTM)"}
        ],
        "supported_cells": ["B0005", "B0006", "B0007", "B0018", "CS2_35", "CS2_36", "CS2_37", "CS2_38"]
    }

@app.get("/battery/{battery_id}", tags=["Diagnostics"])
def get_battery_info(battery_id: str):
    dname = pipeline.get_dataset_for_cell(battery_id)
    q0 = pipeline.get_initial_capacity(battery_id)
    chemistry = "LiCoO2 / Graphite"
    return {
        "battery_id": battery_id,
        "dataset": dname.upper(),
        "chemistry": chemistry,
        "nominal_capacity_ah": q0,
        "eol_capacity_threshold_ah": round(q0 * 0.80, 4)
    }

@app.post("/predict/capacity", response_model=PredictCapacityResponse, tags=["Inference"])
def predict_capacity(req: PredictRequest):
    try:
        cap = pipeline.predict_capacity(
            cell_id=req.battery_id,
            model_type=req.model_type,
            feature_sequence=req.feature_sequence,
            cycle=req.cycle
        )
        return PredictCapacityResponse(
            battery_id=req.battery_id,
            model_type=req.model_type,
            cycle=req.cycle,
            predicted_capacity=round(cap, 4)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/soh", response_model=PredictSOHResponse, tags=["Inference"])
def predict_soh(req: PredictRequest):
    try:
        soh = pipeline.predict_soh(
            cell_id=req.battery_id,
            model_type=req.model_type,
            feature_sequence=req.feature_sequence,
            cycle=req.cycle
        )
        status_label = "Healthy" if soh >= 0.90 else "Degraded" if soh >= 0.80 else "Critical"
        return PredictSOHResponse(
            battery_id=req.battery_id,
            model_type=req.model_type,
            cycle=req.cycle,
            soh=soh,
            status=status_label
        )
    except Exception as e:
        logger.error(f"SOH prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast", response_model=ForecastResponse, tags=["Prognostics"])
def forecast_degradation(req: ForecastRequest):
    try:
        res = pipeline.forecast(
            cell_id=req.battery_id,
            model_type=req.model_type,
            start_cycle=req.start_cycle,
            horizon=req.horizon
        )
        return ForecastResponse(**res)
    except Exception as e:
        logger.error(f"Forecast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/uncertainty", response_model=UncertaintyResponse, tags=["Uncertainty"])
def estimate_uncertainty(req: UncertaintyRequest):
    try:
        res = pipeline.estimate_uncertainty(
            cell_id=req.battery_id,
            feature_sequence=req.feature_sequence,
            n_samples=req.n_samples,
            confidence=req.confidence
        )
        return UncertaintyResponse(**res)
    except Exception as e:
        logger.error(f"Uncertainty estimation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/physics-check", response_model=PhysicsCheckResponse, tags=["Evaluation"])
def check_physics(req: PhysicsCheckRequest):
    try:
        res = pipeline.plausibility_engine.evaluate_plausibility(req.predictions, req.battery_id)
        return PhysicsCheckResponse(
            battery_id=req.battery_id,
            **res
        )
    except Exception as e:
        logger.error(f"Physics check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
