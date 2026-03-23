# Predictive Maintenance — NASA CMAPSS FD001 — Design Spec

**Date:** 2026-03-23
**Project:** Portfolio Project 2 — Industrial Predictive Maintenance
**Status:** Approved (v2 — issues resolved)

---

## 1. Goal

Build a production-quality Remaining Useful Life (RUL) prediction system for industrial turbine engines using the NASA CMAPSS FD001 dataset. The project targets freelance AI/ML portfolio clients in the industrial sector, demonstrating both data science and software engineering maturity.

---

## 2. Dataset

- **Source:** NASA CMAPSS FD001, downloaded via Kaggle API using `scripts/download_data.py`
- **Files used:** `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- **Contents:** 100 turbine engines × N cycles × (21 sensors + 3 operational settings)
- **Target:** RUL (Remaining Useful Life) in cycles
- **Data storage:** `data/raw/` (gitignored), `data/processed/` (gitignored)

---

## 3. Architecture

### 3.1 Folder Structure

```
06 ML Predictive Maintenance/
├── data/
│   ├── raw/                    # original CMAPSS files (gitignored)
│   └── processed/              # engineered features (gitignored)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
├── src/
│   ├── data/
│   │   └── loader.py           # load raw data, engine-based train/val/holdout split
│   ├── features/
│   │   └── engineering.py      # rolling stats, sensor filtering, normalization
│   ├── models/
│   │   ├── lstm_model.py       # PyTorch LSTM architecture
│   │   └── xgboost_model.py    # XGBoost wrapper
│   └── inference/
│       └── pipeline.py         # unified prediction pipeline (loads saved models)
├── app/
│   ├── streamlit_app.py        # entry point
│   ├── tabs/
│   │   ├── operations.py       # Tab 1: technician view
│   │   └── analytics.py        # Tab 2: manager view
│   └── assets/                 # logo, custom CSS
├── scripts/
│   └── download_data.py        # Kaggle API download script
├── models/                     # saved model weights (.pt, .pkl) + scaler (.pkl)
├── docs/
│   └── superpowers/specs/      # design docs
├── .github/
│   └── workflows/
│       └── tests.yml           # CI: run tests on push
├── requirements.txt
├── .gitignore
└── README.md
```

### 3.2 Key Design Principle

Notebooks *consume* `src/` modules — they do not contain reusable logic. The Streamlit app imports `src/inference/pipeline.py` directly. Model changes don't break the dashboard, and the code is reusable beyond the notebooks.

---

## 4. Feature Engineering

**RUL Target (Piecewise Linear with ceiling):**
```python
RUL = min(actual_RUL, 125)
```
Early cycles of a healthy engine have uncertain RUL, so a cap of 125 cycles is applied. Industry-standard approach that improves training stability.

**Features computed in `src/features/engineering.py`:**
1. Remove sensors with near-zero variance across the full training set (typically sensors 1, 5, 6, 10, 16, 18, 19)
2. Rolling mean and std over a **30-cycle window** per sensor, using `min_periods=1` — partial windows at sequence start use available data (no NaN dropped, no backfill)
3. MinMax normalization using a **single scaler fit on the training population** (all training engine rows combined). The same fitted scaler is saved to `models/scaler.pkl` and applied to validation, holdout, and test sets. For new CSV uploads, the training scaler is applied.

**LSTM sequence length:** 30 cycles (matches rolling window). Sequences shorter than 30 cycles (early engine history) use zero-padding on the left.

**Splits (by engine ID — no row-level splitting to prevent leakage):**
- Train: engines 1–80 (80 engines)
- Validation: engines 81–90 (10 engines)
- Local holdout: engines 91–100 (10 engines)
- Official test: `test_FD001.txt` + `RUL_FD001.txt` (100 engines, partial sequences)

---

## 5. Models

### 5.1 LSTM (PyTorch)

- **Input:** sliding window tensor of shape `(batch, 30, N_features)`
- **Architecture:** `LSTM(input_size=N_features, hidden_size=64, num_layers=2, dropout=0.2) → Linear(64→32) → Linear(32→1)`
- **Sequence length:** 30 cycles
- **Loss:** MSE
- **Optimizer:** Adam, lr=1e-3, with ReduceLROnPlateau scheduler (patience=10)
- **Prediction uncertainty:** Monte Carlo Dropout — run 50 forward passes with `model.train()` active, report mean as prediction and std×1.96 as 95% confidence interval
- **Output:** RUL scalar per sequence
- **Saved as:** `models/lstm_model.pt`

### 5.2 XGBoost

- **Input:** flattened feature vector — rolling mean/std of the last 30 cycles for each retained sensor (single row per engine)
- **Hyperparameters:** `n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8`
- **Prediction uncertainty:** bootstrap confidence interval — train 20 models on bootstrap samples, report mean and 5th/95th percentile as the interval
- **Output:** RUL scalar
- **Saved as:** `models/xgboost_model.pkl`

### 5.3 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Square Error (primary ranking metric) |
| MAE | Mean Absolute Error |
| NASA Score | Asymmetric penalty: `sum(exp(-d/13)-1)` for early, `sum(exp(d/10)-1)` for late, where d = predicted - actual |

The model with lower RMSE on the official NASA test set is set as default in the dashboard. User can switch models via sidebar.

---

## 6. Dashboard (Streamlit)

**Deploy target:** Streamlit Cloud (free tier, public GitHub repo)

### Global Layout
- Header with project logo and title
- Sidebar: CSV upload OR use demo data (FD001 test set), engine selector, model selector (LSTM / XGBoost), cost parameters

### Tab 1 — Operaciones (Technician)
- Fleet status table with traffic-light indicator:
  - 🔴 Critical: RUL < 30 cycles
  - 🟡 Warning: 30–60 cycles
  - 🟢 Normal: > 60 cycles
- Sensor degradation line chart for selected engine (cycles on x-axis, sensor value on y-axis)
- RUL prediction bar with confidence interval (shaded region using MC Dropout for LSTM, bootstrap percentiles for XGBoost)
- Export: `st.download_button` downloads a CSV summary (engine ID, predicted RUL, status, confidence interval bounds). PDF export is **out of scope**.

### Tab 2 — Analytics (Plant Manager)
- KPI cards: engines in critical zone, average fleet RUL, estimated cost avoided
- Model comparison table (LSTM vs XGBoost: RMSE, MAE, NASA Score)
- Fleet RUL distribution histogram (Plotly)
- Business case calculator:
  - **User inputs (sidebar):** cost of unplanned failure ($), cost of preventive maintenance ($), production value per hour ($)
  - **Computed savings formula:**
    ```
    engines_saved = count(engines in critical zone)
    savings_per_engine = cost_unplanned - cost_preventive
    total_savings = engines_saved × savings_per_engine
    downtime_hours_recovered = engines_saved × mean_RUL_lead_time × hours_per_cycle
    downtime_value = downtime_hours_recovered × production_value_per_hour
    ```
  - Default values: cost_unplanned=$500,000 / cost_preventive=$50,000 / production_value=$5,000/hr
  - `hours_per_cycle` fixed at 1 hour (typical for CMAPSS, stated in dashboard as assumption)

---

## 7. Business Case (README)

Two quantified KPIs using post-training metric values:
1. **Cost avoided per failure:** "With [mean_RUL_lead_time] cycles advance warning, unplanned failures (~$500K) are replaced by planned maintenance (~$50K), saving ~$450K per event."
2. **Downtime reduction:** "Estimated [N] production hours recovered per year across a fleet of 100 engines, valued at $[X] at $5,000/hr."
   - Formula: `engines_critical × mean_lead_time_cycles × hours_per_cycle × $/hr`
   - All variables are filled post-training with actual model output statistics.

---

## 8. README Structure

```
# 🔧 Predictive Maintenance — NASA CMAPSS FD001
Badges: Python | PyTorch | XGBoost | Streamlit | License | CI

## Business Case
## Live Demo (GIF)
## Architecture Diagram
## Results (metrics table — filled post-training)
## Dataset & Quick Start
## Project Structure
## Contact
```

---

## 9. Reproducibility

```bash
# 1. Clone repo
git clone <repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data (requires ~/.kaggle/kaggle.json)
python scripts/download_data.py

# 4. Run notebooks in order (01 → 04)

# 5. Launch dashboard
streamlit run app/streamlit_app.py
```

---

## 10. CI

`.github/workflows/tests.yml` runs on every push:
- Lint with `flake8`
- Import checks for `src/` modules
- Unit test for `engineering.py`: smoke test with 5 synthetic engines × 50 cycles, assert output shape and no NaN values

---

## 11. Out of Scope

- MLflow tracking (overkill for solo portfolio project)
- FD002/FD003/FD004 subsets (FD001 only for clarity)
- Real-time streaming data (static CSV upload simulates operational use)
- Docker / VPS deployment (Streamlit Cloud handles hosting)
- PDF export (replaced by CSV download button)
