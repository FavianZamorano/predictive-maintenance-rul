# Predictive Maintenance — NASA CMAPSS FD001 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-quality RUL prediction system comparing LSTM (PyTorch) vs XGBoost on NASA CMAPSS FD001, with a two-tab Streamlit dashboard deployed on Streamlit Cloud.

**Architecture:** Modular `src/` package (data → features → models → inference). Notebooks consume `src/` modules. Streamlit app imports `src/inference/pipeline.py`. Single MinMax scaler fit on training data, persisted alongside model weights.

**Tech Stack:** Python 3.10+, PyTorch 2.x, XGBoost 2.x, scikit-learn, pandas, numpy, Streamlit, Plotly, joblib, kaggle (API)

**Spec:** `docs/superpowers/specs/2026-03-23-predictive-maintenance-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `scripts/download_data.py` | Download CMAPSS from Kaggle into `data/raw/` |
| `src/__init__.py` | Package marker |
| `src/data/__init__.py` | Package marker |
| `src/data/loader.py` | Load raw txt files, compute RUL labels, split by engine ID |
| `src/features/__init__.py` | Package marker |
| `src/features/engineering.py` | Rolling stats, sensor filtering, scaler fit/apply |
| `src/models/__init__.py` | Package marker |
| `src/models/lstm_model.py` | PyTorch LSTM architecture + MC Dropout inference |
| `src/models/xgboost_model.py` | XGBoost wrapper with bootstrap confidence intervals |
| `src/inference/__init__.py` | Package marker |
| `src/inference/pipeline.py` | Unified RULPipeline: load models + scaler, predict fleet |
| `tests/test_loader.py` | Unit tests for loader.py |
| `tests/test_engineering.py` | Unit tests for engineering.py (CI smoke test lives here) |
| `tests/test_lstm_model.py` | Unit tests for LSTM forward pass and MC Dropout |
| `tests/test_xgboost_model.py` | Unit tests for XGBoost fit/predict/CI |
| `notebooks/01_EDA.ipynb` | Dataset exploration, sensor variance analysis |
| `notebooks/02_Preprocessing.ipynb` | Feature engineering walkthrough, scaler fitting |
| `notebooks/03_Modeling.ipynb` | Train LSTM and XGBoost, save models |
| `notebooks/04_Evaluation.ipynb` | RMSE, MAE, NASA Score, visual comparisons |
| `app/streamlit_app.py` | Dashboard entry point, layout, sidebar |
| `app/tabs/operations.py` | Tab 1: fleet table, sensor chart, RUL + CI, CSV export |
| `app/tabs/analytics.py` | Tab 2: KPIs, model comparison, histogram, business case |
| `app/assets/style.css` | Custom dark industrial theme |
| `.github/workflows/tests.yml` | CI: flake8 + pytest on push |
| `requirements.txt` | All pinned dependencies |
| `.gitignore` | Exclude data/, models/, __pycache__, .env |
| `README.md` | Professional portfolio README (filled post-training) |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/features/__init__.py`, `src/models/__init__.py`, `src/inference/__init__.py`
- Create: `tests/__init__.py`
- Create: `app/__init__.py`, `app/tabs/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p data/raw data/processed
mkdir -p src/data src/features src/models src/inference
mkdir -p tests
mkdir -p notebooks
mkdir -p app/tabs app/assets
mkdir -p models
mkdir -p scripts
mkdir -p docs
```

- [ ] **Step 2: Create all `__init__.py` files**

```bash
touch src/__init__.py src/data/__init__.py src/features/__init__.py
touch src/models/__init__.py src/inference/__init__.py
touch tests/__init__.py
touch app/__init__.py app/tabs/__init__.py
```

- [ ] **Step 3: Create `requirements.txt`**

```
torch==2.2.2
xgboost==2.0.3
scikit-learn==1.4.2
pandas==2.2.2
numpy==1.26.4
streamlit==1.35.0
plotly==5.22.0
joblib==1.4.2
kaggle==1.6.14
flake8==7.1.0
pytest==8.2.2
```

- [ ] **Step 4: Create `.gitignore`**

```
data/
models/
__pycache__/
*.pyc
.env
*.egg-info/
.DS_Store
.ipynb_checkpoints/
```

- [ ] **Step 5: Verify structure**

```bash
find . -type f | sort
```
Expected: all `__init__.py` files present, `requirements.txt` and `.gitignore` at root.

- [ ] **Step 6: Commit**

```bash
git init
git add requirements.txt .gitignore src/ tests/ app/ scripts/ notebooks/
git commit -m "chore: initial project scaffold"
```

---

## Task 2: Data Download Script

**Files:**
- Create: `scripts/download_data.py`

- [ ] **Step 1: Write `scripts/download_data.py`**

```python
"""Download NASA CMAPSS FD001 dataset via Kaggle API."""
import subprocess
import sys
from pathlib import Path

DATASET = "behrad3d/nasa-cmapss"
OUTPUT_DIR = Path("data/raw")


def download():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} to {OUTPUT_DIR}...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET,
         "-p", str(OUTPUT_DIR), "--unzip"],
        check=True
    )
    print("Download complete. Files in data/raw/:")
    for f in OUTPUT_DIR.iterdir():
        print(f"  {f.name}")


if __name__ == "__main__":
    download()
```

- [ ] **Step 2: Run the script to download data**

```bash
python scripts/download_data.py
```
Expected: `data/raw/` contains `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` (and FD002-004 too — that's fine).

- [ ] **Step 3: Verify files exist**

```bash
ls data/raw/
```
Expected: at least `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` present.

- [ ] **Step 4: Commit**

```bash
git add scripts/download_data.py
git commit -m "feat: add Kaggle data download script"
```

---

## Task 3: Data Loader

**Files:**
- Create: `src/data/loader.py`
- Create: `tests/test_loader.py`

The CMAPSS txt files have no headers. Each row is: `engine_id cycle op1 op2 op3 s1 s2 ... s21 NaN NaN` (space-separated, two trailing NaN columns to drop).

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_loader.py -v
```
Expected: `ImportError` or `ModuleNotFoundError` (loader.py doesn't exist yet).

- [ ] **Step 3: Write `src/data/loader.py`**

```python
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
        df: DataFrame with engine_id column (must have exactly 100 engines for FD001)
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
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_loader.py -v
```
Expected: 6 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/data/loader.py tests/test_loader.py
git commit -m "feat: add CMAPSS data loader with engine-based splits"
```

---

## Task 4: Feature Engineering

**Files:**
- Create: `src/features/engineering.py`
- Create: `tests/test_engineering.py`

- [ ] **Step 1: Write the failing tests**

```python
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
    """CI smoke test: 5 engines × 50 cycles, assert shape + no NaN."""
    df = make_synthetic_df(n_engines=5, cycles_per_engine=50)
    df = add_rolling_features(df)
    feature_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
    assert df[feature_cols].shape == (250, len(feature_cols))
    assert not df[feature_cols].isnull().any().any()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_engineering.py -v
```
Expected: `ImportError` (engineering.py doesn't exist yet).

- [ ] **Step 3: Write `src/features/engineering.py`**

```python
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
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_engineering.py -v
```
Expected: 6 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/features/engineering.py tests/test_engineering.py
git commit -m "feat: add rolling feature engineering and MinMax scaler"
```

---

## Task 5: LSTM Model

**Files:**
- Create: `src/models/lstm_model.py`
- Create: `tests/test_lstm_model.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lstm_model.py
import torch
import pytest
from src.models.lstm_model import LSTMPredictor


def test_forward_pass_shape():
    model = LSTMPredictor(input_size=14)  # 7 sensors × 2 stats
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_lstm_model.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Write `src/models/lstm_model.py`**

```python
"""LSTM model for RUL prediction with Monte Carlo Dropout."""
import numpy as np
import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """Two-layer LSTM with MC Dropout for uncertainty estimation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch,) RUL predictions
        """
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # last timestep
        out = self.relu(self.fc1(self.dropout(out)))
        return self.fc2(out).squeeze(-1)

    def predict_with_ci(
        self, x: torch.Tensor, n_passes: int = 50
    ) -> tuple:
        """Run MC Dropout inference.

        Returns:
            (mean_pred, lower_95, upper_95) as floats
        """
        self.train()  # enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                preds.append(self.forward(x).item())
        mean = float(np.mean(preds))
        std = float(np.std(preds))
        return mean, mean - 1.96 * std, mean + 1.96 * std


def save_model(model: LSTMPredictor, path: str) -> None:
    """Save model weights and config together."""
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config},
        path
    )


def load_model(path: str) -> LSTMPredictor:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model = LSTMPredictor(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_lstm_model.py -v
```
Expected: 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/models/lstm_model.py tests/test_lstm_model.py
git commit -m "feat: add LSTM model with MC Dropout confidence intervals"
```

---

## Task 6: XGBoost Model

**Files:**
- Create: `src/models/xgboost_model.py`
- Create: `tests/test_xgboost_model.py`

- [ ] **Step 1: Write the failing tests**

```python
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
    assert (lower <= pred).all()
    assert (pred <= upper).all()


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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_xgboost_model.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Write `src/models/xgboost_model.py`**

```python
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
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_xgboost_model.py -v
```
Expected: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/models/xgboost_model.py tests/test_xgboost_model.py
git commit -m "feat: add XGBoost model with bootstrap confidence intervals"
```

---

## Task 7: Inference Pipeline

**Files:**
- Create: `src/inference/pipeline.py`

No unit test for the full pipeline (requires saved models). Tested end-to-end in the Evaluation notebook.

- [ ] **Step 1: Write `src/inference/pipeline.py`**

```python
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
```

- [ ] **Step 2: Verify import works**

```bash
python -c "from src.inference.pipeline import RULPipeline; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/inference/pipeline.py
git commit -m "feat: add unified RUL inference pipeline"
```

---

## Task 8: Run All Tests + CI Config

**Files:**
- Create: `.github/workflows/tests.yml`

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v
```
Expected: 13 tests PASSED (loader × 5, engineering × 6, lstm × 4, xgboost × 3... note: adjust if counts differ slightly).

- [ ] **Step 2: Run flake8**

```bash
flake8 src/ --max-line-length=100
```
Expected: no output (no errors).

- [ ] **Step 3: Create `.github/workflows/tests.yml`**

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Lint
        run: flake8 src/ --max-line-length=100

      - name: Run tests
        run: pytest tests/ -v
```

- [ ] **Step 4: Commit**

```bash
git add .github/ tests/
git commit -m "ci: add GitHub Actions workflow for lint and tests"
```

---

## Task 9: Notebook 01 — EDA

**Files:**
- Create: `notebooks/01_EDA.ipynb`

This notebook is exploratory — no tests. Key outputs: sensor variance table, degradation plots, RUL distribution.

- [ ] **Step 1: Create notebook with these cells (in order)**

```python
# Cell 1 — Imports
import sys; sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.loader import load_cmapss_raw, add_rul
```

```python
# Cell 2 — Load data
train, test, rul = load_cmapss_raw()
train = add_rul(train)
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Engines in train: {train['engine_id'].nunique()}")
print(f"Cycles per engine (mean): {train.groupby('engine_id')['cycle'].max().mean():.1f}")
```

```python
# Cell 3 — Sensor variance analysis (identifies which sensors to drop)
sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
variances = train[sensor_cols].var()
print("Sensors with near-zero variance (candidates to drop):")
print(variances[variances < 0.01].sort_values())
```

```python
# Cell 4 — Sensor degradation plot for 3 sample engines
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
retained_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
                    'sensor_9', 'sensor_11', 'sensor_12', 'sensor_14']
sample_engines = [1, 2, 3]
for i, sensor in enumerate(retained_sensors):
    ax = axes[i // 3][i % 3]
    for eng in sample_engines:
        subset = train[train['engine_id'] == eng]
        ax.plot(subset['cycle'], subset[sensor], alpha=0.7, label=f'Engine {eng}')
    ax.set_title(sensor)
    ax.set_xlabel('Cycle')
axes[0][0].legend()
plt.tight_layout()
plt.savefig('../docs/sensor_degradation.png', dpi=100)
plt.show()
```

```python
# Cell 5 — RUL distribution
plt.figure(figsize=(8, 4))
max_cycles = train.groupby('engine_id')['cycle'].max()
plt.hist(max_cycles, bins=20, edgecolor='black')
plt.xlabel('Total engine lifetime (cycles)')
plt.ylabel('Count')
plt.title('Distribution of Engine Lifetimes — FD001')
plt.savefig('../docs/rul_distribution.png', dpi=100)
plt.show()
print(f"Mean lifetime: {max_cycles.mean():.1f} cycles")
print(f"Min: {max_cycles.min()} | Max: {max_cycles.max()}")
```

- [ ] **Step 2: Run all cells in order, verify no errors**

- [ ] **Step 3: Note the sensor list with near-zero variance** — confirm `DROPPED_SENSORS` in `engineering.py` matches. Update if needed.

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_EDA.ipynb docs/sensor_degradation.png docs/rul_distribution.png
git commit -m "docs: add EDA notebook with sensor analysis and RUL distribution"
```

---

## Task 10: Notebook 02 — Preprocessing

**Files:**
- Create: `notebooks/02_Preprocessing.ipynb`

- [ ] **Step 1: Create notebook with these cells**

```python
# Cell 1 — Imports
import sys; sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.loader import load_cmapss_raw, add_rul, split_engines
from src.features.engineering import add_rolling_features, fit_scaler, apply_scaler
```

```python
# Cell 2 — Load and label
train_raw, test_raw, rul_series = load_cmapss_raw()
train_labeled = add_rul(train_raw)
print(f"RUL range: [{train_labeled['RUL'].min()}, {train_labeled['RUL'].max()}]")
```

```python
# Cell 3 — Engine splits
train_df, val_df, holdout_df = split_engines(train_labeled, seed=42)
print(f"Train engines: {train_df['engine_id'].nunique()}")
print(f"Val engines: {val_df['engine_id'].nunique()}")
print(f"Holdout engines: {holdout_df['engine_id'].nunique()}")
```

```python
# Cell 4 — Rolling features
train_feat = add_rolling_features(train_df)
val_feat = add_rolling_features(val_df)
holdout_feat = add_rolling_features(holdout_df)
test_feat = add_rolling_features(test_raw)
feature_cols = [c for c in train_feat.columns if c.endswith('_mean') or c.endswith('_std')]
print(f"Feature count: {len(feature_cols)}")
print(f"Sample features: {feature_cols[:6]}")
```

```python
# Cell 5 — Fit scaler on train, apply to all
scaler = fit_scaler(train_feat, model_dir='../models')
train_scaled = apply_scaler(train_feat, scaler)
val_scaled = apply_scaler(val_feat, scaler)
holdout_scaled = apply_scaler(holdout_feat, scaler)
test_scaled = apply_scaler(test_feat, scaler)
print("Scaler saved to models/scaler.pkl")
print(f"Train scaled shape: {train_scaled.shape}")
```

```python
# Cell 6 — Save processed data
import os; os.makedirs('../data/processed', exist_ok=True)
train_scaled.to_parquet('../data/processed/train.parquet', index=False)
val_scaled.to_parquet('../data/processed/val.parquet', index=False)
holdout_scaled.to_parquet('../data/processed/holdout.parquet', index=False)
test_scaled.to_parquet('../data/processed/test.parquet', index=False)
# Save RUL ground truth for test
rul_series.to_frame('RUL').to_parquet('../data/processed/test_rul.parquet', index=False)
print("Processed data saved.")
```

- [ ] **Step 2: Run all cells in order, verify no errors**

- [ ] **Step 3: Verify `data/processed/` contains 5 parquet files and `models/scaler.pkl` exists**

- [ ] **Step 4: Commit**

```bash
git add notebooks/02_Preprocessing.ipynb
git commit -m "docs: add preprocessing notebook with feature engineering walkthrough"
```

---

## Task 11: Notebook 03 — Modeling

**Files:**
- Create: `notebooks/03_Modeling.ipynb`

This is the longest notebook. It trains both models and saves weights.

- [ ] **Step 1: Create the LSTM training section**

```python
# Cell 1 — Imports
import sys; sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.lstm_model import LSTMPredictor, save_model
```

```python
# Cell 2 — Dataset class for sliding windows
class CMAPSSDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len=30):
        self.seq_len = seq_len
        self.sequences = []
        self.targets = []
        for _, group in df.groupby('engine_id'):
            X = group[feature_cols].values.astype(np.float32)
            y = group['RUL'].values.astype(np.float32)
            for i in range(len(X)):
                if i < seq_len:
                    pad = np.zeros((seq_len - i - 1, X.shape[1]), dtype=np.float32)
                    seq = np.vstack([pad, X[:i+1]])
                else:
                    seq = X[i-seq_len+1:i+1]
                self.sequences.append(seq)
                self.targets.append(y[i])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.tensor(self.targets[idx])
```

```python
# Cell 3 — Load processed data
feature_cols = [c for c in pd.read_parquet('../data/processed/train.parquet').columns
                if c.endswith('_mean') or c.endswith('_std')]
train_df = pd.read_parquet('../data/processed/train.parquet')
val_df = pd.read_parquet('../data/processed/val.parquet')

train_ds = CMAPSSDataset(train_df, feature_cols)
val_ds = CMAPSSDataset(val_df, feature_cols)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
print(f"Input features: {len(feature_cols)}")
```

```python
# Cell 4 — Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMPredictor(input_size=len(feature_cols)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion = nn.MSELoss()

best_val_loss = float('inf')
history = {'train': [], 'val': []}

for epoch in range(100):
    model.train()
    total_train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    train_loss = total_train_loss / len(train_loader)

    model.eval()
    with torch.no_grad():
        val_loss = sum(
            criterion(model(xb.to(device)), yb.to(device)).item()
            for xb, yb in val_loader
        ) / len(val_loader)

    scheduler.step(val_loss)
    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model(model, '../models/lstm_model.pt')

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print(f"\nBest Val Loss: {best_val_loss:.4f}")
print("Model saved to models/lstm_model.pt")
```

```python
# Cell 5 — Training curve
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot(history['train'], label='Train')
plt.plot(history['val'], label='Validation')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
plt.title('LSTM Training Curve')
plt.legend()
plt.savefig('../docs/lstm_training_curve.png', dpi=100)
plt.show()
```

- [ ] **Step 2: Create the XGBoost training section**

```python
# Cell 6 — XGBoost: prepare tabular features (last row per engine)
from src.models.xgboost_model import XGBoostPredictor

def get_last_row_per_engine(df, feature_cols):
    last_rows = df.groupby('engine_id').last().reset_index()
    X = last_rows[feature_cols].values
    y = last_rows['RUL'].values
    return X, y

X_train, y_train = get_last_row_per_engine(train_df, feature_cols)
X_val, y_val = get_last_row_per_engine(val_df, feature_cols)
print(f"XGBoost train shape: {X_train.shape}")
```

```python
# Cell 7 — Train XGBoost
xgb_model = XGBoostPredictor()
xgb_model.fit(X_train, y_train)
xgb_model.save('../models/xgboost_model.pkl')
val_preds = xgb_model.predict(X_val)
val_rmse = np.sqrt(np.mean((val_preds - y_val)**2))
print(f"XGBoost Val RMSE: {val_rmse:.2f}")
print("Model saved to models/xgboost_model.pkl")
```

```python
# Cell 8 — Feature importance plot
import matplotlib.pyplot as plt
importances = xgb_model.get_feature_importance()
top_idx = np.argsort(importances)[-15:]
plt.figure(figsize=(8, 6))
plt.barh([feature_cols[i] for i in top_idx], importances[top_idx])
plt.xlabel('Importance')
plt.title('XGBoost Top 15 Feature Importances')
plt.tight_layout()
plt.savefig('../docs/xgb_feature_importance.png', dpi=100)
plt.show()
```

- [ ] **Step 3: Run all cells in order**

- [ ] **Step 4: Verify `models/` contains `lstm_model.pt`, `xgboost_model.pkl`, `scaler.pkl`**

```bash
ls models/
```

- [ ] **Step 5: Commit**

```bash
git add notebooks/03_Modeling.ipynb docs/lstm_training_curve.png docs/xgb_feature_importance.png
git commit -m "docs: add modeling notebook training LSTM and XGBoost"
```

---

## Task 12: Notebook 04 — Evaluation

**Files:**
- Create: `notebooks/04_Evaluation.ipynb`

- [ ] **Step 1: Create notebook with evaluation cells**

```python
# Cell 1 — Imports and helpers
import sys; sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from src.inference.pipeline import RULPipeline
from src.data.loader import load_cmapss_raw

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d/13) - 1, np.exp(d/10) - 1)
    return float(np.sum(score))
```

```python
# Cell 2 — Load official test set
_, test_raw, rul_ground_truth = load_cmapss_raw()
y_test = rul_ground_truth.values
print(f"Test engines: {test_raw['engine_id'].nunique()} | Ground truth RUL values: {len(y_test)}")
```

```python
# Cell 3 — LSTM predictions on test set
# Note: for official test set, each engine's last row is the prediction point
pipeline_lstm = RULPipeline(model_dir='../models', model_type='lstm')

# Use last cycle per engine for prediction
test_last = test_raw.groupby('engine_id').last().reset_index()
# But pipeline needs full sequences — use full test_raw
results_lstm = pipeline_lstm.predict(test_raw, n_mc=50)
y_pred_lstm = results_lstm.set_index('engine_id')['rul_pred'].values

lstm_rmse = rmse(y_test, y_pred_lstm)
lstm_mae = mae(y_test, y_pred_lstm)
lstm_nasa = nasa_score(y_test, y_pred_lstm)
print(f"LSTM  — RMSE: {lstm_rmse:.2f} | MAE: {lstm_mae:.2f} | NASA Score: {lstm_nasa:.0f}")
```

```python
# Cell 4 — XGBoost predictions on test set
pipeline_xgb = RULPipeline(model_dir='../models', model_type='xgboost')
results_xgb = pipeline_xgb.predict(test_raw)
y_pred_xgb = results_xgb.set_index('engine_id')['rul_pred'].values

xgb_rmse = rmse(y_test, y_pred_xgb)
xgb_mae = mae(y_test, y_pred_xgb)
xgb_nasa = nasa_score(y_test, y_pred_xgb)
print(f"XGBoost — RMSE: {xgb_rmse:.2f} | MAE: {xgb_mae:.2f} | NASA Score: {xgb_nasa:.0f}")
```

```python
# Cell 5 — Summary table
import pandas as pd
summary = pd.DataFrame({
    'Model': ['LSTM', 'XGBoost'],
    'RMSE': [lstm_rmse, xgb_rmse],
    'MAE': [lstm_mae, xgb_mae],
    'NASA Score': [lstm_nasa, xgb_nasa],
})
summary['Best'] = summary['RMSE'] == summary['RMSE'].min()
print(summary.to_string(index=False))
# Save for README
summary.to_csv('../docs/metrics_summary.csv', index=False)
```

```python
# Cell 6 — Prediction vs Actual scatter plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, title in [(axes[0], y_pred_lstm, 'LSTM'), (axes[1], y_pred_xgb, 'XGBoost')]:
    ax.scatter(y_test, preds, alpha=0.5, s=20)
    lims = [0, max(y_test.max(), preds.max())]
    ax.plot(lims, lims, 'r--', label='Perfect')
    ax.set_xlabel('True RUL'); ax.set_ylabel('Predicted RUL')
    ax.set_title(title); ax.legend()
plt.tight_layout()
plt.savefig('../docs/prediction_scatter.png', dpi=100)
plt.show()
```

- [ ] **Step 2: Run all cells in order**

- [ ] **Step 3: Record final metrics** (RMSE, MAE, NASA Score for both models) — you will need these for the README in Task 16.

- [ ] **Step 4: Commit**

```bash
git add notebooks/04_Evaluation.ipynb docs/metrics_summary.csv docs/prediction_scatter.png
git commit -m "docs: add evaluation notebook with RMSE, MAE, NASA Score comparison"
```

---

## Task 13: Dashboard Foundation + Tab 1 (Operations)

**Files:**
- Create: `app/streamlit_app.py`
- Create: `app/tabs/operations.py`
- Create: `app/assets/style.css`

- [ ] **Step 1: Create `app/assets/style.css`**

```css
/* Industrial dark theme */
.stApp {
    background-color: #0e1117;
}
.metric-card {
    background: #1e2530;
    border-radius: 8px;
    padding: 1rem;
    border-left: 4px solid #00d4aa;
}
.status-critical { color: #ff4b4b; font-weight: bold; }
.status-warning  { color: #ffa500; font-weight: bold; }
.status-normal   { color: #00d4aa; font-weight: bold; }
```

- [ ] **Step 2: Create `app/tabs/operations.py`**

```python
"""Tab 1: Operations view for maintenance technicians."""
import pandas as pd
import plotly.express as px
import streamlit as st


STATUS_EMOJI = {"critical": "🔴", "warning": "🟡", "normal": "🟢"}
STATUS_LABEL = {"critical": "CRITICAL (<30)", "warning": "WARNING (30-60)", "normal": "NORMAL (>60)"}


def render(results: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    """
    Args:
        results: output of RULPipeline.predict() — engine_id, rul_pred, ci_lower, ci_upper, status
        raw_df: original sensor dataframe for degradation charts
    """
    st.header("Fleet Status")

    # --- Fleet table ---
    display = results.copy()
    display["Status"] = display["status"].map(
        lambda s: f"{STATUS_EMOJI[s]} {STATUS_LABEL[s]}"
    )
    display = display.rename(columns={
        "engine_id": "Engine", "rul_pred": "RUL (cycles)",
        "ci_lower": "CI Lower", "ci_upper": "CI Upper",
    })
    st.dataframe(
        display[["Engine", "RUL (cycles)", "CI Lower", "CI Upper", "Status"]],
        use_container_width=True,
        hide_index=True,
    )

    # --- Engine selector ---
    engine_id = st.selectbox(
        "Select engine for detail view",
        options=sorted(results["engine_id"].unique()),
    )
    engine_data = raw_df[raw_df["engine_id"] == engine_id].sort_values("cycle")
    engine_result = results[results["engine_id"] == engine_id].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted RUL", f"{engine_result['rul_pred']:.0f} cycles")
    with col2:
        status = engine_result["status"]
        st.markdown(
            f"**Status:** <span class='status-{status}'>{STATUS_EMOJI[status]} {status.upper()}</span>",
            unsafe_allow_html=True,
        )
    st.caption(
        f"95% Confidence Interval: [{engine_result['ci_lower']:.0f}, {engine_result['ci_upper']:.0f}] cycles"
    )

    # --- Sensor degradation chart ---
    st.subheader("Sensor Degradation")
    retained_sensors = [c for c in engine_data.columns if c.startswith("sensor_")][:6]
    selected_sensor = st.selectbox("Select sensor", retained_sensors)
    fig = px.line(
        engine_data, x="cycle", y=selected_sensor,
        title=f"Engine {engine_id} — {selected_sensor} over time",
        labels={"cycle": "Cycle", selected_sensor: "Sensor Value"},
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- CSV export ---
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export Fleet Report (CSV)",
        csv,
        file_name="fleet_rul_report.csv",
        mime="text/csv",
    )
```

- [ ] **Step 3: Create `app/streamlit_app.py`**

```python
"""Predictive Maintenance Dashboard — entry point."""
import sys
from pathlib import Path

# Allow importing src/ from app/
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from src.data.loader import load_cmapss_raw
from src.inference.pipeline import RULPipeline
from app.tabs import operations, analytics

st.set_page_config(
    page_title="Industrial RUL Predictor",
    page_icon="🔧",
    layout="wide",
)

# Load custom CSS
with open(Path(__file__).parent / "assets" / "style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔧 RUL Predictor")
    st.caption("NASA CMAPSS FD001")
    st.divider()

    uploaded = st.file_uploader("Upload sensor CSV", type=["csv", "txt"])
    model_choice = st.selectbox("Model", ["LSTM", "XGBoost"])

    st.subheader("Cost Parameters")
    cost_failure = st.number_input("Unplanned failure cost ($)", value=500_000, step=10_000)
    cost_preventive = st.number_input("Preventive maintenance cost ($)", value=50_000, step=1_000)
    prod_value = st.number_input("Production value ($/hr)", value=5_000, step=500)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_demo_data():
    _, test_raw, _ = load_cmapss_raw(data_dir="data/raw")
    return test_raw

if uploaded:
    raw_df = pd.read_csv(uploaded, sep=r"\s+", header=None)
    from src.data.loader import COLUMNS
    raw_df = raw_df.iloc[:, :len(COLUMNS)]
    raw_df.columns = COLUMNS
else:
    raw_df = load_demo_data()

# ── Run inference ─────────────────────────────────────────────────────────────
@st.cache_data
def run_inference(model_type: str, _df: pd.DataFrame):
    pipeline = RULPipeline(model_dir="models", model_type=model_type.lower())
    return pipeline.predict(_df)

model_type = "lstm" if model_choice == "LSTM" else "xgboost"
results = run_inference(model_type, raw_df)

# ── Tabs ──────────────────────────────────────────────────────────────────────
st.title("🔧 Industrial RUL Predictor")
st.caption("Remaining Useful Life prediction for turbine engine fleets")

tab1, tab2 = st.tabs(["⚙️ Operaciones", "📊 Analytics"])

with tab1:
    operations.render(results, raw_df)

with tab2:
    analytics.render(results, cost_failure, cost_preventive, prod_value)
```

- [ ] **Step 4: Verify app launches**

```bash
streamlit run app/streamlit_app.py
```
Expected: browser opens, sidebar visible, Tab 1 shows fleet table (may show an error if models/ not populated yet — that's OK at this stage).

- [ ] **Step 5: Commit**

```bash
git add app/
git commit -m "feat: add Streamlit dashboard foundation and operations tab"
```

---

## Task 14: Tab 2 — Analytics

**Files:**
- Create: `app/tabs/analytics.py`

- [ ] **Step 1: Create `app/tabs/analytics.py`**

```python
"""Tab 2: Analytics view for plant managers."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render(
    results: pd.DataFrame,
    cost_failure: float,
    cost_preventive: float,
    prod_value_per_hour: float,
) -> None:
    """
    Args:
        results: RULPipeline.predict() output
        cost_failure: cost of one unplanned failure ($)
        cost_preventive: cost of planned maintenance ($)
        prod_value_per_hour: production revenue ($/hr)
    """
    critical = results[results["status"] == "critical"]
    warning = results[results["status"] == "warning"]
    avg_rul = results["rul_pred"].mean()

    # Business case computation
    engines_at_risk = len(critical)
    savings_per_engine = cost_failure - cost_preventive
    total_savings = engines_at_risk * savings_per_engine
    mean_lead_time = critical["rul_pred"].mean() if engines_at_risk > 0 else 0
    downtime_hours = engines_at_risk * mean_lead_time * 1.0  # 1 hr/cycle assumption
    downtime_value = downtime_hours * prod_value_per_hour

    # ── KPI Cards ────────────────────────────────────────────────────────────
    st.header("Fleet Analytics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔴 Critical Engines", engines_at_risk)
    col2.metric("🟡 Warning Engines", len(warning))
    col3.metric("Avg Fleet RUL", f"{avg_rul:.0f} cycles")
    col4.metric("Est. Cost Avoided", f"${total_savings:,.0f}")

    st.divider()

    # ── Model comparison ─────────────────────────────────────────────────────
    st.subheader("Model Performance (Official Test Set)")
    try:
        metrics = pd.read_csv("docs/metrics_summary.csv")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.info("Run notebook 04_Evaluation.ipynb to generate metrics.")

    # ── RUL distribution ────────────────────────────────────────────────────
    st.subheader("Fleet RUL Distribution")
    fig = px.histogram(
        results, x="rul_pred", nbins=20,
        color="status",
        color_discrete_map={"critical": "#ff4b4b", "warning": "#ffa500", "normal": "#00d4aa"},
        template="plotly_dark",
        labels={"rul_pred": "Predicted RUL (cycles)", "count": "Engines"},
        title="Distribution of Remaining Useful Life Across Fleet",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Business case ────────────────────────────────────────────────────────
    st.subheader("Business Case Calculator")
    st.caption("Assumptions: 1 cycle = 1 hour of operation")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cost Savings from Early Detection**")
        fig_savings = go.Figure(go.Bar(
            x=["Unplanned Failure", "Preventive Maintenance", "Net Savings"],
            y=[cost_failure * engines_at_risk, cost_preventive * engines_at_risk, total_savings],
            marker_color=["#ff4b4b", "#ffa500", "#00d4aa"],
        ))
        fig_savings.update_layout(template="plotly_dark", yaxis_title="Cost ($)")
        st.plotly_chart(fig_savings, use_container_width=True)

    with col2:
        st.markdown("**Production Value Recovered**")
        st.metric("Downtime hours avoided", f"{downtime_hours:.0f} hrs")
        st.metric("Production value recovered", f"${downtime_value:,.0f}")
        st.caption(
            f"Based on {engines_at_risk} critical engines × "
            f"{mean_lead_time:.0f} cycles advance notice × "
            f"${ prod_value_per_hour:,.0f}/hr"
        )
```

- [ ] **Step 2: Run app and verify both tabs render**

```bash
streamlit run app/streamlit_app.py
```

- [ ] **Step 3: Commit**

```bash
git add app/tabs/analytics.py
git commit -m "feat: add analytics tab with KPIs, model comparison, and business case"
```

---

## Task 15: README

**Files:**
- Create: `README.md`

Fill in the metrics table with actual values from `docs/metrics_summary.csv` after running notebook 04.

- [ ] **Step 1: Create `README.md`**

```markdown
# 🔧 Predictive Maintenance — NASA CMAPSS FD001

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![CI](https://github.com/<YOUR_USERNAME>/<YOUR_REPO>/actions/workflows/tests.yml/badge.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Predict the **Remaining Useful Life (RUL)** of industrial turbine engines before failure occurs.
> Comparing LSTM (PyTorch) vs XGBoost with a two-tab industrial dashboard.

---

## Business Case

Unplanned turbofan engine failures cost the aviation and industrial sector an estimated **$500K+ per event** in emergency repairs, unscheduled downtime, and safety incidents.

This system provides **two quantified KPIs:**

| KPI | Value |
|-----|-------|
| Cost avoided per failure | ~$450K (replace emergency repair with planned maintenance) |
| Production hours recovered | _N_ hrs/yr across 100-engine fleet at $5K/hr = $_X_K |

> _Fill in actual values from notebook 04 after training._

---

## Live Demo

[**🚀 Try the dashboard →**](https://your-streamlit-url.streamlit.app)

<!-- Add dashboard GIF here after recording -->

---

## Results

| Model   | RMSE  | MAE   | NASA Score |
|---------|-------|-------|------------|
| LSTM    | _TBD_ | _TBD_ | _TBD_      |
| XGBoost | _TBD_ | _TBD_ | _TBD_      |

> _Fill in from `docs/metrics_summary.csv` after running notebook 04._

---

## Architecture

```
Raw Data → Feature Engineering → LSTM / XGBoost → RULPipeline → Streamlit Dashboard
              (rolling stats,                        (MC Dropout CI /
               MinMax scaler)                         Bootstrap CI)
```

```
src/
├── data/loader.py       # load + RUL labels + engine splits
├── features/engineering.py  # rolling stats, scaler
├── models/lstm_model.py      # PyTorch LSTM + MC Dropout
├── models/xgboost_model.py   # XGBoost + bootstrap CI
└── inference/pipeline.py    # unified predict API
```

---

## Dataset

NASA CMAPSS FD001 — 100 turbine engines, 21 sensors, simulated degradation to failure.
Downloaded automatically via Kaggle API.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
cd <YOUR_REPO>

# 2. Install
pip install -r requirements.txt

# 3. Download data (requires ~/.kaggle/kaggle.json)
python scripts/download_data.py

# 4. Run notebooks in order
jupyter notebook notebooks/

# 5. Launch dashboard
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
├── data/raw/               # CMAPSS files (gitignored)
├── notebooks/              # EDA → Preprocessing → Modeling → Evaluation
├── src/                    # Reusable ML package
├── app/                    # Streamlit dashboard
├── models/                 # Saved weights (gitignored)
├── scripts/                # Data download
└── docs/                   # Plots, metrics
```

---

## Contact

Built by **[Your Name]** · [LinkedIn](https://linkedin.com/in/yourprofile) · [Portfolio](https://yoursite.com)
```

- [ ] **Step 2: Replace `<YOUR_USERNAME>/<YOUR_REPO>` with actual values once repo is created on GitHub**

- [ ] **Step 3: After running notebook 04, fill in the `_TBD_` values in the Results table from `docs/metrics_summary.csv`**

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add professional README with business case and quick start"
```

---

## Task 16: Final Checks + Streamlit Cloud Deploy

- [ ] **Step 1: Run full test suite one final time**

```bash
pytest tests/ -v
```
Expected: all tests PASSED.

- [ ] **Step 2: Run flake8**

```bash
flake8 src/ app/ scripts/ --max-line-length=100 --exclude=__pycache__
```
Expected: no errors.

- [ ] **Step 3: Create GitHub repo and push**

```bash
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
git branch -M main
git push -u origin main
```

- [ ] **Step 4: Bundle model files for Streamlit Cloud**

  `models/` is gitignored — Streamlit Cloud has no access to local files. Before deploying, commit the trained model files:

  ```bash
  # Temporarily remove models/ from .gitignore for this commit only
  git add -f models/lstm_model.pt models/xgboost_model.pkl models/scaler.pkl
  git commit -m "chore: add trained model artifacts for Streamlit Cloud deploy"
  ```

  Then restore the .gitignore entry after deploy (models/ stays ignored for future local development).

- [ ] **Step 5: Deploy to Streamlit Cloud**
  - Go to [share.streamlit.io](https://share.streamlit.io)
  - Connect GitHub repo
  - Set main file: `app/streamlit_app.py`
  - Add secrets if needed (none required for this project)
  - Click Deploy

- [ ] **Step 6: Update README with live URL and CI badge**

- [ ] **Step 7: Record dashboard GIF** (use LICEcap or similar) and add to README

- [ ] **Step 8: Final commit**

```bash
git add README.md
git commit -m "docs: add live demo URL and dashboard GIF"
git push
```

---

## Summary

| Task | Deliverable |
|------|-------------|
| 1 | Project scaffold |
| 2 | `scripts/download_data.py` |
| 3 | `src/data/loader.py` + tests |
| 4 | `src/features/engineering.py` + tests |
| 5 | `src/models/lstm_model.py` + tests |
| 6 | `src/models/xgboost_model.py` + tests |
| 7 | `src/inference/pipeline.py` |
| 8 | CI workflow |
| 9–12 | 4 notebooks (EDA → Evaluation) |
| 13 | Dashboard + Tab 1 (Operations) |
| 14 | Tab 2 (Analytics) |
| 15 | README.md |
| 16 | Deploy to Streamlit Cloud |
