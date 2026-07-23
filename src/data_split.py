"""
Step 3 — Data Splitting
The one rule that matters more than anything else here: split by
ENGINE ID, never by row. If even one cycle from an engine leaks into
both train and validation, the model has effectively already seen
that engine's degradation pattern and your validation score becomes
fiction.

The real test set (test_FD001_features.csv) is already a separate
holdout provided by NASA -- it stays untouched until final evaluation
and is NOT part of this split.
"""

from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def split_by_engine(df: pd.DataFrame, val_size: float = 0.2, random_state: int = 42):
    """
    Split a dataframe into train/val by engine_id, not by row.
    Returns (train_df, val_df).
    """
    engine_ids = df["engine_id"].unique()
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(engine_ids)

    n_val = max(1, int(len(shuffled) * val_size))
    val_ids = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])

    train_df = df[df["engine_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["engine_id"].isin(val_ids)].reset_index(drop=True)

    return train_df, val_df


def assert_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Hard safety check: fail loudly if any engine_id appears in both sets."""
    train_ids = set(train_df["engine_id"].unique())
    val_ids = set(val_df["engine_id"].unique())
    overlap = train_ids & val_ids
    assert not overlap, f"LEAKAGE DETECTED: engine_ids in both train and val: {overlap}"
    print(f"No leakage confirmed. Train engines: {len(train_ids)}, Val engines: {len(val_ids)}")


def run(processed_dir=None, subset="FD001", val_size: float = 0.2, random_state: int = 42):
    processed_dir = Path(processed_dir) if processed_dir else REPO_ROOT / "data" / "processed"

    full_df = pd.read_csv(processed_dir / f"train_{subset}_features.csv")
    print(f"Full training data: {full_df.shape}, engines: {full_df['engine_id'].nunique()}")

    train_df, val_df = split_by_engine(full_df, val_size=val_size, random_state=random_state)
    assert_no_leakage(train_df, val_df)

    print(f"Train split: {train_df.shape}")
    print(f"Val split:   {val_df.shape}")

    train_df.to_csv(processed_dir / f"train_{subset}_split.csv", index=False)
    val_df.to_csv(processed_dir / f"val_{subset}_split.csv", index=False)
    print(f"Saved to {processed_dir}/train_{subset}_split.csv and val_{subset}_split.csv")

    # Reminder: the real test set is a separate, untouched holdout.
    test_path = processed_dir / f"test_{subset}_features.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        print(f"(Untouched holdout test set: {test_df.shape}, engines: {test_df['engine_id'].nunique()})")

    return train_df, val_df


if __name__ == "__main__":
    run()
