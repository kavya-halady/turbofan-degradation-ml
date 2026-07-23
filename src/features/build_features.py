"""
Step 2 — Feature Engineering
Raw sensor snapshots don't tell a model much on their own -- what
actually signals an approaching failure is the TREND. This module adds,
per engine, per sensor:
  - rolling mean / std over a window of cycles (smooths noise, reveals drift)
  - rate of change (slope) over the same window (how fast it's moving)
  - cumulative delta from the engine's first reading (total drift so far)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[2]
NON_FEATURE_COLS = {"engine_id", "cycle", "RUL"}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Every column that isn't an id/label is a raw feature (settings + sensors)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def add_rolling_features(df: pd.DataFrame, feature_cols: list[str], window: int = 5) -> pd.DataFrame:
    """Rolling mean & std per engine, per feature. min_periods=1 avoids NaNs
    at the start of each engine's history (early cycles just use what's available)."""
    df = df.sort_values(["engine_id", "cycle"]).copy()
    grouped = df.groupby("engine_id")[feature_cols]

    roll_mean = grouped.transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    roll_std = grouped.transform(lambda s: s.rolling(window=window, min_periods=1).std()).fillna(0)

    roll_mean.columns = [f"{c}_roll_mean{window}" for c in feature_cols]
    roll_std.columns = [f"{c}_roll_std{window}" for c in feature_cols]

    return pd.concat([df, roll_mean, roll_std], axis=1)


def _slope(series: pd.Series) -> float:
    """Linear regression slope of a small window -- 'how fast is this sensor moving'."""
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    return float(np.polyfit(x, series.values, 1)[0])


def add_rate_of_change(df: pd.DataFrame, feature_cols: list[str], window: int = 5) -> pd.DataFrame:
    """Rolling slope per engine, per feature, over the same window size."""
    df = df.sort_values(["engine_id", "cycle"]).copy()
    slope_cols = {}
    for col in feature_cols:
        slope_cols[f"{col}_slope{window}"] = (
            df.groupby("engine_id")[col]
            .transform(lambda s: s.rolling(window=window, min_periods=2).apply(_slope, raw=False))
            .fillna(0)
        )
    return pd.concat([df, pd.DataFrame(slope_cols, index=df.index)], axis=1)


def add_cumulative_drift(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Total drift from each engine's very first reading -- captures long-run degradation."""
    df = df.sort_values(["engine_id", "cycle"]).copy()
    first_vals = df.groupby("engine_id")[feature_cols].transform("first")
    drift = df[feature_cols] - first_vals
    drift.columns = [f"{c}_cum_drift" for c in feature_cols]
    return pd.concat([df, drift], axis=1)


def report_pca_variance(df: pd.DataFrame, feature_cols: list[str], n_components: int = 5) -> None:
    """Informational only: shows how much variance a handful of PCA components
    would capture. With ~18 sensors this is rarely worth trading away
    interpretability for, especially with tree-based models -- but good to see."""
    pca = PCA(n_components=min(n_components, len(feature_cols)))
    pca.fit(df[feature_cols].fillna(0))
    explained = pca.explained_variance_ratio_
    print(f"PCA — variance explained by first {len(explained)} components:")
    for i, v in enumerate(explained, 1):
        print(f"  PC{i}: {v:.3f}  (cumulative: {explained[:i].sum():.3f})")


def build_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    raw_feature_cols = get_feature_columns(df)
    df = add_rolling_features(df, raw_feature_cols, window)
    df = add_rate_of_change(df, raw_feature_cols, window)
    df = add_cumulative_drift(df, raw_feature_cols)
    return df


def run(processed_dir=None, subset="FD001", window: int = 5):
    processed_dir = Path(processed_dir) if processed_dir else REPO_ROOT / "data" / "processed"

    train_df = pd.read_csv(processed_dir / f"train_{subset}_clean.csv")
    test_df = pd.read_csv(processed_dir / f"test_{subset}_clean.csv")

    raw_feature_cols = get_feature_columns(train_df)
    print(f"Base features: {len(raw_feature_cols)}")

    report_pca_variance(train_df, raw_feature_cols)

    train_feat = build_features(train_df, window=window)
    test_feat = build_features(test_df, window=window)

    print(f"Train features shape: {train_feat.shape}  (added {train_feat.shape[1] - train_df.shape[1]} new columns)")
    print(f"Test features shape:  {test_feat.shape}")
    print(f"Any NaNs in train? {train_feat.isna().any().any()}")
    print(f"Any NaNs in test?  {test_feat.isna().any().any()}")

    train_feat.to_csv(processed_dir / f"train_{subset}_features.csv", index=False)
    test_feat.to_csv(processed_dir / f"test_{subset}_features.csv", index=False)
    print(f"Saved to {processed_dir}/train_{subset}_features.csv and test_{subset}_features.csv")

    return train_feat, test_feat


if __name__ == "__main__":
    run()
