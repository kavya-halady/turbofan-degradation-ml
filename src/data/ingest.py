"""
Step 1a — Data Collection
Loads the raw C-MAPSS train/test/RUL text files and turns them into
tidy, properly-labeled pandas DataFrames.

Raw files are whitespace-separated with no header and a couple of
trailing blank columns caused by trailing spaces in the original
NASA export — we drop those.
"""

from pathlib import Path
import pandas as pd

# Column layout is fixed across all C-MAPSS sub-datasets (FD001-FD004)
INDEX_COLUMNS = ["engine_id", "cycle"]
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


def load_cmapss_file(path: Path) -> pd.DataFrame:
    """Load a single raw train_* or test_* C-MAPSS txt file."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    # Raw files sometimes have 2 trailing all-NaN columns from stray
    # trailing whitespace in the original export -- drop anything
    # beyond the 26 real columns.
    df = df.iloc[:, : len(ALL_COLUMNS)]
    df.columns = ALL_COLUMNS
    return df


def load_rul_file(path: Path) -> pd.DataFrame:
    """Load the RUL_*.txt file (ground-truth RUL for the test set)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"])
    # RUL file has one row per engine, in engine_id order starting at 1
    df["engine_id"] = df.index + 1
    return df[["engine_id", "RUL"]]


def load_dataset(raw_dir: str | Path, subset: str = "FD001"):
    """
    Load train, test, and RUL files for a given C-MAPSS subset.

    Example:
        train_df, test_df, rul_df = load_dataset("data/raw", "FD001")
    """
    raw_dir = Path(raw_dir)
    train_df = load_cmapss_file(raw_dir / f"train_{subset}.txt")
    test_df = load_cmapss_file(raw_dir / f"test_{subset}.txt")
    rul_df = load_rul_file(raw_dir / f"RUL_{subset}.txt")
    return train_df, test_df, rul_df


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    train_df, test_df, rul_df = load_dataset(repo_root / "data" / "raw", "FD001")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("RUL shape:", rul_df.shape)
    print(train_df.head())
