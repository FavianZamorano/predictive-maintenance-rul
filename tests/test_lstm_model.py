"""Tests for LSTM model with MC Dropout."""
import torch
import pytest
from src.models.lstm_model import LSTMPredictor


def test_forward_pass_shape():
    model = LSTMPredictor(input_size=14)  # 7 sensors x 2 stats
    x = torch.randn(8, 30, 14)  # batch=8, seq=30, features=14
    out = model(x)
    assert out.shape == (8,)


def test_mc_dropout_variability():
    """MC Dropout should return different predictions across passes."""
    model = LSTMPredictor(input_size=14)
    x = torch.randn(1, 30, 14)
    model.train()  # enable dropout
    preds = [model(x).item() for _ in range(20)]
    # Not all identical (dropout is active)
    assert len(set(round(p, 6) for p in preds)) > 1


def test_predict_with_ci_shape():
    model = LSTMPredictor(input_size=14)
    x = torch.randn(1, 30, 14)
    mean, lower, upper = model.predict_with_ci(x, n_passes=10)
    assert isinstance(mean, float)
    assert lower <= mean <= upper


def test_output_is_scalar_per_sample():
    model = LSTMPredictor(input_size=14)
    x = torch.randn(4, 30, 14)
    out = model(x)
    assert out.shape == (4,)
