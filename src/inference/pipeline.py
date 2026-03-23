"""Unified inference pipeline for RUL prediction."""
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.features.engineering import add_rolling_features, apply_scaler
from src.models.lstm_model import load_model as load_lstm
from src.models.xgboost_model import XGBoostPredictor

SEQ_LEN = 30


class RULPipeline:
    """Load trained models and predict RUL for a fleet dataframe.

    Args:
        model_dir: directory containing lstm_model.pt, xgboost_model.pkl, scaler.pkl
        model_type: 'lstm' or 'xgboost'
    """

    def __init__(self, model_dir: str = "models", model_type: str = "lstm"):
        self.model_dir = Path(model_dir)
        self.model_type = model_type
        self.scaler = joblib.load(self.model_dir / "scaler.pkl")

        if model_type == "lstm":
            self.model = load_lstm(str(self.model_dir / "lstm_model.pt"))
        elif model_type == "xgboost":
            self.model = XGBoostPredictor.load(
                str(self.model_dir / "xgboost_model.pkl")
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def predict(self, df: pd.DataFrame, n_mc: int = 50) -> pd.DataFrame:
        """Predict RUL for each engine in df.

        Args:
            df: raw sensor DataFrame with columns matching CMAPSS COLUMNS
            n_mc: number of MC Dropout passes (LSTM only)

        Returns:
            DataFrame with columns: engine_id, rul_pred, ci_lower, ci_upper, status
        """
        df = add_rolling_features(df)
        df = apply_scaler(df, self.scaler)
        feature_cols = [
            c for c in df.columns if c.endswith("_mean") or c.endswith("_std")
        ]

        results = []
        for engine_id, group in df.groupby("engine_id"):
            X = group[feature_cols].values

            if self.model_type == "lstm":
                pred, lower, upper = self._predict_lstm(X, feature_cols, n_mc)
            else:
                pred, lower, upper = self._predict_xgboost(X)

            status = self._get_status(pred)
            results.append({
                "engine_id": engine_id,
                "rul_pred": round(float(pred), 1),
                "ci_lower": round(float(max(0, lower)), 1),
                "ci_upper": round(float(upper), 1),
                "status": status,
            })

        return pd.DataFrame(results).sort_values("rul_pred")

    def _predict_lstm(self, X: np.ndarray, feature_cols: list, n_mc: int) -> tuple:
        if len(X) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(X), X.shape[1]))
            X = np.vstack([pad, X])
        else:
            X = X[-SEQ_LEN:]
        tensor = torch.FloatTensor(X).unsqueeze(0)
        return self.model.predict_with_ci(tensor, n_passes=n_mc)

    def _predict_xgboost(self, X: np.ndarray) -> tuple:
        last_row = X[[-1]]
        pred, lower, upper = self.model.predict_with_ci(last_row)
        return pred[0], lower[0], upper[0]

    @staticmethod
    def _get_status(rul: float) -> str:
        if rul < 30:
            return "critical"
        elif rul < 60:
            return "warning"
        return "normal"
