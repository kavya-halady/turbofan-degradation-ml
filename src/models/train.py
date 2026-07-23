"""
Step 4 — Model Selection & Training
Trains a Linear Regression baseline (sanity check) and an XGBoost
regressor (primary model) to predict Remaining Useful Life (RUL).
Both runs are logged to MLflow; the better model (by validation RMSE)
is saved to disk for Step 5 (evaluation) and Step 6 (serving).
"""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame, target: str, id_columns: list[str]) -> list[str]:
    exclude = set(id_columns) | {target}
    return [c for c in df.columns if c not in exclude]


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_baseline(X_train, y_train, X_val, y_val) -> tuple[LinearRegression, dict]:
    model = LinearRegression()
    model.fit(X_train, y_train)
    metrics = evaluate(y_val, model.predict(X_val))
    return model, metrics


def train_xgboost(X_train, y_train, X_val, y_val, params: dict) -> tuple[XGBRegressor, dict]:
    model = XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=params["random_state"],
        eval_metric="rmse",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    metrics = evaluate(y_val, model.predict(X_val))
    return model, metrics


def run(config_path: str = "config.yaml"):
    config_path = REPO_ROOT / config_path if not Path(config_path).is_absolute() else Path(config_path)
    cfg = load_config(config_path)

    processed_dir = REPO_ROOT / cfg["data"]["processed_dir"]
    target = cfg["target"]
    id_columns = cfg["id_columns"]

    train_df = pd.read_csv(processed_dir / cfg["data"]["train_file"])
    val_df = pd.read_csv(processed_dir / cfg["data"]["val_file"])

    feature_cols = get_feature_columns(train_df, target, id_columns)
    print(f"Training on {len(feature_cols)} features, {len(train_df)} train rows, {len(val_df)} val rows")

    X_train, y_train = train_df[feature_cols], train_df[target]
    X_val, y_val = val_df[feature_cols], val_df[target]

    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    results = {}

    with mlflow.start_run(run_name="baseline_linear_regression"):
        baseline_model, baseline_metrics = train_baseline(X_train, y_train, X_val, y_val)
        mlflow.log_metrics(baseline_metrics)
        mlflow.sklearn.log_model(baseline_model, name="model")
        results["baseline"] = (baseline_model, baseline_metrics)
        print("Baseline (Linear Regression):", baseline_metrics)

    with mlflow.start_run(run_name="xgboost_regressor"):
        xgb_params = cfg["model"]["xgboost"]
        mlflow.log_params(xgb_params)
        xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val, xgb_params)
        mlflow.log_metrics(xgb_metrics)
        mlflow.xgboost.log_model(xgb_model, name="model")
        results["xgboost"] = (xgb_model, xgb_metrics)
        print("XGBoost:", xgb_metrics)

    # Pick the winner by validation RMSE (lower is better)
    best_name = min(results, key=lambda k: results[k][1]["rmse"])
    best_model, best_metrics = results[best_name]
    print(f"\nBest model: {best_name}  (RMSE={best_metrics['rmse']:.2f}, MAE={best_metrics['mae']:.2f}, R2={best_metrics['r2']:.3f})")

    output_dir = REPO_ROOT / cfg["model"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output_dir / "model.joblib")
    with open(output_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(output_dir / "model_info.json", "w") as f:
        json.dump({"model_type": best_name, "metrics": best_metrics}, f, indent=2)

    print(f"Saved best model ({best_name}) to {output_dir}/model.joblib")
    return best_model, best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
