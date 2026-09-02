from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "online"
    service: str = "Battery Twin REST API"
    version: str = "0.1.0"

class PredictRequest(BaseModel):
    battery_id: str = Field(...)
    model_type: str = Field("pinn", description="Options: physics, ai, pinn")
    cycle: Optional[int] = Field(None)
    feature_sequence: Optional[List[List[float]]] = Field(
        None,
        description="5x6 sequence of features [cycle, voltage_mean, current_mean, temp_mean, temp_max, capacity_prev]"
    )

class PredictCapacityResponse(BaseModel):
    battery_id: str
    model_type: str
    cycle: Optional[int]
    predicted_capacity: float
    unit: str = "Ah"

class PredictSOHResponse(BaseModel):
    battery_id: str
    model_type: str
    cycle: Optional[int]
    soh: float
    status: str

class ForecastRequest(BaseModel):
    battery_id: str = Field(...)
    model_type: str = Field("pinn")
    start_cycle: int = Field(100)
    horizon: int = Field(30)

class ForecastResponse(BaseModel):
    battery_id: str
    model_type: str
    start_cycle: int
    horizon: int
    cycles: List[int]
    predicted_capacity: List[float]
    predicted_soh: List[float]
    plausibility: Dict[str, Any]

class UncertaintyRequest(BaseModel):
    battery_id: str = Field(...)
    feature_sequence: List[List[float]] = Field(..., description="5x6 sequence")
    n_samples: int = Field(100)
    confidence: float = Field(0.90)

class UncertaintyResponse(BaseModel):
    battery_id: str
    mean_capacity: float
    std_capacity: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    mean_soh: float

class PhysicsCheckRequest(BaseModel):
    battery_id: str = Field(...)
    predictions: List[float] = Field(..., description="Array of capacity predictions")

class PhysicsCheckResponse(BaseModel):
    battery_id: str
    status: str
    plausibility_score: float
    monotonicity_violations: int
    negative_capacity_violations: int
    capacity_jump_violations: int
    explanations: List[str]
