"""XGBoost model with bootstrap confidence intervals."""
import joblib
import numpy as np
import xgboost as xgb


class XGBoostPredictor:
    """XGBoost regressor with bootstrap ensemble for uncertainty estimation."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        n_bootstrap: int = 20,
        seed: int = 42,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=seed,
        )
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.model = xgb.XGBRegressor(**self.params)
        self.bootstrap_models: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostPredictor":
        self.model.fit(X, y)
        rng = np.random.default_rng(self.seed)
        self.bootstrap_models = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, len(X), size=len(X))
            m = xgb.XGBRegressor(**self.params)
            m.fit(X[idx], y[idx])
            self.bootstrap_models.append(m)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_with_ci(self, X: np.ndarray) -> tuple:
        """Return (predictions, lower_5th_pct, upper_95th_pct)."""
        pred = self.predict(X)
        bootstrap_preds = np.array([m.predict(X) for m in self.bootstrap_models])
        lower = np.percentile(bootstrap_preds, 5, axis=0)
        upper = np.percentile(bootstrap_preds, 95, axis=0)
        return pred, lower, upper

    def get_feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "XGBoostPredictor":
        return joblib.load(path)
