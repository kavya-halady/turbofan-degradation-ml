"""
Step 5b — Hyperparameter Tuning with Optuna
Searches XGBoost's hyperparameter space, scoring each trial by 3-fold
grouped cross-validation (grouped by engine_id, same leakage rule as
everywhere else in this pipeline). The best configuration is retrained
on the full training pool and only replaces the currently saved model
if it actually beats it on the untouched holdout test set.
"""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import standard_metrics  # reuse RMSE/MAE/R2/NASA-score logic


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
    }


def cv_rmse(df: pd.DataFrame, feature_cols: list[str], target: str, params: dict, n_splits: int = 3) -> float:
    gkf = GroupKFold(n_splits=n_splits)
    X, y, groups = df[feature_cols], df[target], df["engine_id"]
    rmses = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        model = XGBRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        rmses.append(standard_metrics(y.iloc[val_idx], preds)["rmse"])
    return float(np.mean(rmses))


def run(config_path: str = "config.yaml", n_trials: int = 30):
    config_path = REPO_ROOT / config_path if not Path(config_path).is_absolute() else Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = REPO_ROOT / cfg["data"]["processed_dir"]
    target = cfg["target"]
    model_dir = REPO_ROOT / cfg["model"]["output_dir"]

    feature_cols = json.loads((model_dir / "feature_columns.json").read_text())
    train_df = pd.read_csv(processed_dir / cfg["data"]["train_file"])
    val_df = pd.read_csv(processed_dir / cfg["data"]["val_file"])
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    test_df = pd.read_csv(processed_dir / cfg["data"]["test_file"])

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        return cv_rmse(full_train_df, feature_cols, target, params)

    print(f"Running Optuna search: {n_trials} trials, 3-fold grouped CV each...")
    study = optuna.create_study(direction="minimize", study_name="turbineiq-xgb-tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\nBest trial CV RMSE: {study.best_value:.2f}")
    print("Best params:", study.best_params)

    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    best_params = {**study.best_params, "random_state": 42}

    with mlflow.start_run(run_name="optuna_best_xgboost"):
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_rmse", study.best_value)

        # Retrain on the FULL training pool with the best params found
        final_model = XGBRegressor(**best_params)
        final_model.fit(full_train_df[feature_cols], full_train_df[target])

        tuned_test_metrics = standard_metrics(test_df[target], final_model.predict(test_df[feature_cols]))
        mlflow.log_metrics({f"test_{k}": v for k, v in tuned_test_metrics.items()})
        mlflow.xgboost.log_model(final_model, name="model")

    print("\nTuned model performance on holdout test set:")
    print(f"  RMSE={tuned_test_metrics['rmse']:.2f}  MAE={tuned_test_metrics['mae']:.2f}  "
          f"R2={tuned_test_metrics['r2']:.3f}  NASA={tuned_test_metrics['nasa_score']:.1f}")

    # Compare against the currently saved model -- only replace if actually better
    current_model = joblib.load(model_dir / "model.joblib")
    current_test_metrics = standard_metrics(test_df[target], current_model.predict(test_df[feature_cols]))
    print(f"\nCurrent saved model on holdout test set:")
    print(f"  RMSE={current_test_metrics['rmse']:.2f}  MAE={current_test_metrics['mae']:.2f}  "
          f"R2={current_test_metrics['r2']:.3f}  NASA={current_test_metrics['nasa_score']:.1f}")

    if tuned_test_metrics["rmse"] < current_test_metrics["rmse"]:
        joblib.dump(final_model, model_dir / "model.joblib")
        with open(model_dir / "model_info.json", "w") as f:
            json.dump({"model_type": "xgboost_tuned", "params": best_params,
                       "metrics": tuned_test_metrics}, f, indent=2)
        with open(model_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\n✅ Tuned model IMPROVED holdout RMSE ({current_test_metrics['rmse']:.2f} -> "
              f"{tuned_test_metrics['rmse']:.2f}). Saved as new models/model.joblib.")
    else:
        with open(model_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\n⚠️  Tuned model did NOT beat the current saved model on holdout RMSE "
              f"({tuned_test_metrics['rmse']:.2f} vs {current_test_metrics['rmse']:.2f}). "
              f"Keeping existing model.joblib. Best params still saved to best_params.json for reference.")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()
    run(args.config, args.n_trials)
