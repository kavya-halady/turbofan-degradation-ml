"""
Step 5 — Model Evaluation & Optimization
- Implements the NASA/PHM08 asymmetric scoring function: predicting a
  failure too LATE is penalized far more heavily than predicting it
  too early, matching the real cost structure of maintenance decisions.
- Runs 5-fold cross-validation grouped by engine_id (never split a
  single engine's cycles across folds).
- Evaluates the saved model against the true, untouched holdout test set.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]


def nasa_score(y_true, y_pred) -> float:
    """
    PHM08 / NASA scoring function (Saxena & Goebel, 2008).
    d = predicted - actual
      d < 0 (early prediction): score = exp(-d/13) - 1   (lighter penalty)
      d >= 0 (late prediction): score = exp(d/10) - 1     (heavier penalty)
    Lower total score is better. Unlike RMSE, this reflects that a late
    RUL prediction is operationally worse than an early one -- an
    engine flagged too late risks failure before maintenance happens.
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


def standard_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "nasa_score": nasa_score(y_true, y_pred),
    }


def grouped_cross_validate(df: pd.DataFrame, feature_cols: list[str], target: str,
                            xgb_params: dict, n_splits: int = 5) -> dict:
    """5-fold CV grouped by engine_id -- no engine ever appears in both
    the train and validation portion of the same fold."""
    gkf = GroupKFold(n_splits=n_splits)
    X, y, groups = df[feature_cols], df[target], df["engine_id"]

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        model = XGBRegressor(
            n_estimators=xgb_params["n_estimators"],
            max_depth=xgb_params["max_depth"],
            learning_rate=xgb_params["learning_rate"],
            subsample=xgb_params["subsample"],
            colsample_bytree=xgb_params["colsample_bytree"],
            random_state=xgb_params["random_state"],
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        m = standard_metrics(y.iloc[val_idx], preds)
        fold_metrics.append(m)
        print(f"  Fold {fold}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  "
              f"R2={m['r2']:.3f}  NASA={m['nasa_score']:.1f}")

    avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    std = {k: float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    return {"avg": avg, "std": std, "folds": fold_metrics}


def run(config_path: str = "config.yaml"):
    config_path = REPO_ROOT / config_path if not Path(config_path).is_absolute() else Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = REPO_ROOT / cfg["data"]["processed_dir"]
    target = cfg["target"]
    model_dir = REPO_ROOT / cfg["model"]["output_dir"]

    model = joblib.load(model_dir / "model.joblib")
    feature_cols = json.loads((model_dir / "feature_columns.json").read_text())

    # --- 1. Evaluate on the real, untouched holdout test set ---
    test_df = pd.read_csv(processed_dir / cfg["data"]["test_file"])
    test_metrics = standard_metrics(test_df[target], model.predict(test_df[feature_cols]))
    print("=== Holdout Test Set Performance ===")
    print(f"RMSE={test_metrics['rmse']:.2f}  MAE={test_metrics['mae']:.2f}  "
          f"R2={test_metrics['r2']:.3f}  NASA Score={test_metrics['nasa_score']:.1f}")

    # --- 2. Grouped 5-fold cross-validation on the full training pool ---
    # Combine train+val splits back together for CV -- the real test set
    # stays completely separate and untouched throughout.
    train_df = pd.read_csv(processed_dir / cfg["data"]["train_file"])
    val_df = pd.read_csv(processed_dir / cfg["data"]["val_file"])
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"\n=== 5-Fold Grouped Cross-Validation ({full_train_df['engine_id'].nunique()} engines) ===")
    cv_results = grouped_cross_validate(full_train_df, feature_cols, target, cfg["model"]["xgboost"])
    print(f"\nCV Average: RMSE={cv_results['avg']['rmse']:.2f} (+/-{cv_results['std']['rmse']:.2f})  "
          f"MAE={cv_results['avg']['mae']:.2f}  R2={cv_results['avg']['r2']:.3f}  "
          f"NASA={cv_results['avg']['nasa_score']:.1f}")

    # Save everything for the record
    report = {"holdout_test": test_metrics, "cross_validation": cv_results}
    with open(model_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved evaluation report to {model_dir}/evaluation_report.json")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
