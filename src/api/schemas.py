"""
Pydantic schemas for the TurbineIQ prediction API.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CycleReading(BaseModel):
    """One engine cycle's raw sensor + operational setting readings.
    Keys must match the model's expected base columns (e.g. 'sensor_2',
    'setting_1') -- see GET /model-info for the exact list."""
    values: Dict[str, float]


class PredictRULRequest(BaseModel):
    engine_id: int = Field(..., description="Identifier for the engine being monitored")
    cycles: List[CycleReading] = Field(
        ...,
        min_length=1,
        description="Chronological list of recent cycle readings, OLDEST first. "
                     "At least 5 cycles recommended so rolling/slope features are meaningful.",
    )


class PredictRULResponse(BaseModel):
    engine_id: int
    predicted_rul: float = Field(..., description="Predicted remaining useful life, in cycles")
    cycles_used: int


class PredictFailureRiskRequest(PredictRULRequest):
    threshold_cycles: int = Field(30, description="Flag as high-risk if predicted RUL falls at or below this many cycles")


class PredictFailureRiskResponse(BaseModel):
    engine_id: int
    predicted_rul: float
    high_risk: bool
    threshold_cycles: int
    cycles_used: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_type: str
    metrics: Dict[str, float]
    required_base_columns: List[str]
    n_engineered_features: int
