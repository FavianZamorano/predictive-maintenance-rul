# tests/test_loader.py
import numpy as np
import pandas as pd
import pytest
from src.data.loader import COLUMNS, load_cmapss_raw, add_rul, split_engines


def make_synthetic_df(n_engines=5, cycles_per_engine=50):
    """Create synthetic CMAPSS-like dataframe."""
    rows = []
    for eng in range(1, n_engines + 1):
        for cyc in range(1, cycles_per_engine + 1):
            row = [eng, cyc, 0.0, 0.0, 0.0] + list(np.random.rand(21))
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def test_columns_length():
    assert len(COLUMNS) == 26  # engine_id + cycle + 3 op + 21 sensors


def test_add_rul_max_cap():
    df = make_synthetic_df(n_engines=2, cycles_per_engine=200)
    df = add_rul(df, max_rul=125)
    assert df["RUL"].max() == 125


def test_add_rul_last_cycle_is_zero():
    df = make_synthetic_df(n_engines=3, cycles_per_engine=50)
    df = add_rul(df, max_rul=125)
    last_cycles = df.groupby("engine_id")["cycle"].max()
    for eng in df["engine_id"].unique():
        last_row = df[(df["engine_id"] == eng) & (df["cycle"] == last_cycles[eng])]
        assert last_row["RUL"].values[0] == 0


def test_split_engines_by_id():
    df = make_synthetic_df(n_engines=10, cycles_per_engine=30)
    train, val, holdout = split_engines(df, seed=42)
    # No engine appears in more than one split
    train_ids = set(train["engine_id"].unique())
    val_ids = set(val["engine_id"].unique())
    holdout_ids = set(holdout["engine_id"].unique())
    assert len(train_ids & val_ids) == 0
    assert len(train_ids & holdout_ids) == 0
    assert len(val_ids & holdout_ids) == 0


def test_split_engines_covers_all():
    df = make_synthetic_df(n_engines=10, cycles_per_engine=30)
    train, val, holdout = split_engines(df, seed=42)
    all_ids = set(df["engine_id"].unique())
    covered = set(train["engine_id"].unique()) | set(val["engine_id"].unique()) | set(holdout["engine_id"].unique())
    assert covered == all_ids


def test_split_proportions():
    df = make_synthetic_df(n_engines=100, cycles_per_engine=10)
    train, val, holdout = split_engines(df, seed=42)
    assert len(train["engine_id"].unique()) == 80
    assert len(val["engine_id"].unique()) == 10
    assert len(holdout["engine_id"].unique()) == 10
