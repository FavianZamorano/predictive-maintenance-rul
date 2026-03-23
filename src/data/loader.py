"""Load and split NASA CMAPSS FD001 data."""
import numpy as np
import pandas as pd
from pathlib import Path

COLUMNS = (
    ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_cmapss_raw(data_dir: str = "data/raw", subset: str = "FD001") -> tuple:
    """Load train, test CSVs and ground-truth RUL for given subset.

    Returns:
        train_df, test_df, rul_series
    """
    data_dir = Path(data_dir)

    def _read(fname):
        df = pd.read_csv(
            data_dir / fname, sep=r"\s+", header=None, engine="python"
        )
        # Drop trailing NaN columns (last 2 cols in raw files)
        df = df.iloc[:, : len(COLUMNS)]
        df.columns = COLUMNS
        return df

    train = _read(f"train_{subset}.txt")
    test = _read(f"test_{subset}.txt")

    rul_raw = pd.read_csv(
        data_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None, engine="python"
    )
    rul = rul_raw.iloc[:, 0].reset_index(drop=True)
    rul.name = "RUL"

    return train, test, rul


def add_rul(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """Compute RUL for each row and apply ceiling cap."""
    df = df.copy()
    max_cycles = df.groupby("engine_id")["cycle"].max()
    df["RUL"] = df["engine_id"].map(max_cycles) - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=max_rul)
    return df


def split_engines(
    df: pd.DataFrame, seed: int = 42
) -> tuple:
    """Split by engine ID: 80% train / 10% val / 10% holdout.

    Args:
        df: DataFrame with engine_id column
        seed: random seed for reproducibility

    Returns:
        train_df, val_df, holdout_df
    """
    engines = np.array(sorted(df["engine_id"].unique()))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(engines)
    n = len(shuffled)
    cut1 = int(0.8 * n)
    cut2 = int(0.9 * n)
    return (
        df[df["engine_id"].isin(shuffled[:cut1])].copy(),
        df[df["engine_id"].isin(shuffled[cut1:cut2])].copy(),
        df[df["engine_id"].isin(shuffled[cut2:])].copy(),
    )
