"""Feature engineering for CMAPSS dataset."""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Sensors constant across FD001 (near-zero variance — confirmed in EDA notebook)
DROPPED_SENSORS = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]
WINDOW = 30


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return sensor column names that are NOT dropped."""
    return [
        c for c in df.columns
        if c.startswith("sensor_") and c not in DROPPED_SENSORS
    ]


def add_rolling_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Add rolling mean and std for each retained sensor.

    Uses min_periods=1 so no NaN is introduced at sequence boundaries.
    Std at period=1 is set to 0.
    """
    df = df.copy().sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    feature_cols = get_feature_columns(df)

    for col in feature_cols:
        grp = df.groupby("engine_id")[col]
        df[f"{col}_mean"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{col}_std"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0.0)
        )
    return df


def fit_scaler(df: pd.DataFrame, model_dir: str = "models") -> MinMaxScaler:
    """Fit MinMaxScaler on rolling feature columns and save to disk."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    feature_cols = [
        c for c in df.columns if c.endswith("_mean") or c.endswith("_std")
    ]
    scaler = MinMaxScaler()
    scaler.fit(df[feature_cols])
    joblib.dump(scaler, Path(model_dir) / "scaler.pkl")
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    """Apply a pre-fitted scaler to rolling feature columns."""
    df = df.copy()
    feature_cols = [
        c for c in df.columns if c.endswith("_mean") or c.endswith("_std")
    ]
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df
