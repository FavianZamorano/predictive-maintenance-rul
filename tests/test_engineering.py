# tests/test_engineering.py
import numpy as np
import pandas as pd
import pytest
from src.data.loader import COLUMNS, add_rul
from src.features.engineering import (
    DROPPED_SENSORS,
    WINDOW,
    add_rolling_features,
    fit_scaler,
    apply_scaler,
    get_feature_columns,
)


def make_synthetic_df(n_engines=5, cycles_per_engine=50, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for eng in range(1, n_engines + 1):
        for cyc in range(1, cycles_per_engine + 1):
            row = [eng, cyc, 0.0, 0.0, 0.0] + list(rng.random(21))
            rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS)
    return add_rul(df)


def test_no_nan_after_rolling(tmp_path):
    df = make_synthetic_df()
    df = add_rolling_features(df)
    feature_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
    assert not df[feature_cols].isnull().any().any()


def test_rolling_output_shape():
    df = make_synthetic_df(n_engines=5, cycles_per_engine=50)
    n_rows = len(df)
    df = add_rolling_features(df)
    assert len(df) == n_rows  # no rows dropped


def test_dropped_sensors_absent():
    df = make_synthetic_df()
    feature_cols = get_feature_columns(df)
    for dropped in DROPPED_SENSORS:
        assert dropped not in feature_cols


def test_scaler_fit_and_apply(tmp_path):
    df = make_synthetic_df()
    df = add_rolling_features(df)
    scaler = fit_scaler(df, model_dir=str(tmp_path))
    df_scaled = apply_scaler(df, scaler)
    feature_cols = [c for c in df_scaled.columns if c.endswith("_mean") or c.endswith("_std")]
    # After MinMax scaling, all values in [0, 1]
    assert df_scaled[feature_cols].min().min() >= -1e-6
    assert df_scaled[feature_cols].max().max() <= 1 + 1e-6


def test_scaler_saved_to_disk(tmp_path):
    import joblib
    df = make_synthetic_df()
    df = add_rolling_features(df)
    fit_scaler(df, model_dir=str(tmp_path))
    assert (tmp_path / "scaler.pkl").exists()
    loaded = joblib.load(tmp_path / "scaler.pkl")
    assert hasattr(loaded, "transform")


def test_ci_smoke_test():
    """CI smoke test: 5 engines x 50 cycles, assert shape + no NaN."""
    df = make_synthetic_df(n_engines=5, cycles_per_engine=50)
    df = add_rolling_features(df)
    feature_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
    assert df[feature_cols].shape == (250, len(feature_cols))
    assert not df[feature_cols].isnull().any().any()
