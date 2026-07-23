"""
Step 1b — Preprocessing
- Computes the RUL label for the training set
- Drops sensors with zero variance (they carry no signal)
- Normalizes sensor readings, fitting the scaler on TRAIN ONLY
- Saves the cleaned train/test sets and the fitted scaler to disk
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ingest import load_dataset, SENSOR_COLUMNS, SETTING_COLUMNS

RUL_CLIP = 125  # standard C-MAPSS trick: engines behave "healthy" for
                # a long stretch, so cap RUL rather than let the model
                # try to distinguish 300 vs 280 cycles left


def compute_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """Add a RUL column to the training set: RUL = max_cycle - cycle."""
    max_cycle = train_df.groupby("engine_id")["cycle"].transform("max")
    train_df = train_df.copy()
    train_df["RUL"] = (max_cycle - train_df["cycle"]).clip(upper=RUL_CLIP)
    return train_df


def add_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach ground-truth RUL to the test set.
    The test set only has the LAST recorded cycle per engine to score
    against — RUL_df gives the true remaining life at that final cycle.
    """
    last_cycle = test_df.groupby("engine_id")["cycle"].transform("max")
    test_df = test_df.copy()
    test_df["cycles_from_end"] = last_cycle - test_df["cycle"]
    merged = test_df.merge(rul_df, on="engine_id", how="left")
    merged["RUL"] = (merged["RUL"] + merged["cycles_from_end"]).clip(upper=RUL_CLIP)
    return merged.drop(columns=["cycles_from_end"])


def drop_zero_variance(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Drop sensor columns that are constant in the training data."""
    variances = train_df[SENSOR_COLUMNS].var()
    zero_var_cols = variances[variances == 0].index.tolist()
    print(f"Dropping {len(zero_var_cols)} zero-variance sensors: {zero_var_cols}")

    kept_sensors = [c for c in SENSOR_COLUMNS if c not in zero_var_cols]
    train_df = train_df.drop(columns=zero_var_cols)
    test_df = test_df.drop(columns=zero_var_cols)
    return train_df, test_df, kept_sensors


def normalize(train_df, test_df, feature_cols, scaler_path: Path):
    """Fit a MinMaxScaler on TRAIN ONLY, transform both sets, save the scaler."""
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted scaler to {scaler_path}")

    return train_df, test_df


# Repo root = two levels up from this file (src/data/preprocess.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]


def run(raw_dir=None, processed_dir=None, subset="FD001"):
    raw_dir = Path(raw_dir) if raw_dir else REPO_ROOT / "data" / "raw"
    processed_dir = Path(processed_dir) if processed_dir else REPO_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, rul_df = load_dataset(raw_dir, subset)

    train_df = compute_rul(train_df)
    test_df = add_test_rul(test_df, rul_df)

    train_df, test_df, kept_sensors = drop_zero_variance(train_df, test_df)
    feature_cols = SETTING_COLUMNS + kept_sensors

    train_df, test_df = normalize(
        train_df, test_df, feature_cols, processed_dir / "scaler.joblib"
    )

    train_df.to_csv(processed_dir / f"train_{subset}_clean.csv", index=False)
    test_df.to_csv(processed_dir / f"test_{subset}_clean.csv", index=False)

    print("Train (clean):", train_df.shape)
    print("Test (clean):", test_df.shape)
    return train_df, test_df


if __name__ == "__main__":
    run()
