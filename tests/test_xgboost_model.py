# tests/test_xgboost_model.py
import numpy as np
import pytest
from src.models.xgboost_model import XGBoostPredictor


def make_data(n=200, n_features=28):
    rng = np.random.default_rng(0)
    X = rng.random((n, n_features))
    y = rng.integers(0, 125, size=n).astype(float)
    return X, y


def test_fit_and_predict_shape():
    X, y = make_data()
    model = XGBoostPredictor()
    model.fit(X, y)
    preds = model.predict(X[:10])
    assert preds.shape == (10,)


def test_predict_with_ci_bounds():
    X, y = make_data()
    model = XGBoostPredictor()
    model.fit(X, y)
    pred, lower, upper = model.predict_with_ci(X[:5])
    assert pred.shape == (5,)
    # Check that CI bounds are valid (lower <= upper)
    assert (lower <= upper).all()


def test_save_load(tmp_path):
    X, y = make_data()
    model = XGBoostPredictor()
    model.fit(X, y)
    save_path = str(tmp_path / "xgb.pkl")
    model.save(save_path)
    loaded = XGBoostPredictor.load(save_path)
    np.testing.assert_array_almost_equal(
        model.predict(X[:5]), loaded.predict(X[:5])
    )
