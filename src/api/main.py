"""
Step 6 — Model Deployment
FastAPI service exposing RUL prediction and failure-risk endpoints.

Design note: the trained model expects ENGINEERED features (rolling
means, slopes, cumulative drift -- see Step 2), not raw sensor values.
A real client sends a recent window of raw cycle readings for one
engine; this service reuses the exact same feature-engineering logic
from src/features/build_features.py to compute the engineered features
on the fly, then predicts from the most recent row. This mirrors how
the model was trained, so there's no train/serve skew.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "features"))
from build_features import build_features  # noqa: E402

from .schemas import (  # noqa: E402
    CycleReading,
    HealthResponse,
    ModelInfoResponse,
    PredictFailureRiskRequest,
    PredictFailureRiskResponse,
    PredictRULRequest,
    PredictRULResponse,
)

app = FastAPI(title="TurbineIQ API", description="Predictive maintenance for turbofan engines")

# --- Loaded once at startup, not per-request ---
_state = {"model": None, "feature_cols": None, "base_cols": None, "model_info": None}

_ENGINEERED_SUFFIXES = ("_roll_mean5", "_roll_std5", "_slope5", "_cum_drift")


@app.on_event("startup")
def load_model():
    with open(REPO_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    model_dir = REPO_ROOT / cfg["model"]["output_dir"]

    _state["model"] = joblib.load(model_dir / "model.joblib")
    _state["feature_cols"] = json.loads((model_dir / "feature_columns.json").read_text())
    _state["base_cols"] = [
        c for c in _state["feature_cols"] if not c.endswith(_ENGINEERED_SUFFIXES)
    ]
    info_path = model_dir / "model_info.json"
    _state["model_info"] = json.loads(info_path.read_text()) if info_path.exists() else {}
    print(f"Model loaded. Expecting {len(_state['base_cols'])} base columns, "
          f"{len(_state['feature_cols'])} engineered features.")


def _cycles_to_feature_row(engine_id: int, cycles: list[CycleReading]) -> pd.Series:
    """Turn a list of raw cycle readings into the engineered feature row
    for the MOST RECENT cycle, using the same logic as Step 2 training."""
    base_cols = _state["base_cols"]

    rows = []
    for reading in cycles:
        missing = set(base_cols) - set(reading.values.keys())
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required fields in cycle reading: {sorted(missing)}",
            )
        rows.append({c: reading.values[c] for c in base_cols})

    df = pd.DataFrame(rows)
    df["engine_id"] = engine_id
    df["cycle"] = range(1, len(df) + 1)

    engineered = build_features(df, window=5)
    latest_row = engineered.iloc[[-1]]

    feature_cols = _state["feature_cols"]
    missing_engineered = set(feature_cols) - set(latest_row.columns)
    if missing_engineered:
        raise HTTPException(
            status_code=500,
            detail=f"Internal feature mismatch, missing: {sorted(missing_engineered)}",
        )
    return latest_row[feature_cols]


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=_state["model"] is not None)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    info = _state["model_info"]
    return ModelInfoResponse(
        model_type=info.get("model_type", "unknown"),
        metrics=info.get("metrics", {}),
        required_base_columns=_state["base_cols"],
        n_engineered_features=len(_state["feature_cols"]),
    )


@app.post("/predict/rul", response_model=PredictRULResponse)
def predict_rul(request: PredictRULRequest):
    X = _cycles_to_feature_row(request.engine_id, request.cycles)
    pred = float(_state["model"].predict(X)[0])
    pred = max(0.0, pred)  # RUL can't be negative
    return PredictRULResponse(
        engine_id=request.engine_id,
        predicted_rul=round(pred, 1),
        cycles_used=len(request.cycles),
    )


@app.post("/predict/failure-risk", response_model=PredictFailureRiskResponse)
def predict_failure_risk(request: PredictFailureRiskRequest):
    X = _cycles_to_feature_row(request.engine_id, request.cycles)
    pred = float(_state["model"].predict(X)[0])
    pred = max(0.0, pred)
    return PredictFailureRiskResponse(
        engine_id=request.engine_id,
        predicted_rul=round(pred, 1),
        high_risk=pred <= request.threshold_cycles,
        threshold_cycles=request.threshold_cycles,
        cycles_used=len(request.cycles),
    )
