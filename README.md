# 🔧 Predictive Maintenance — NASA CMAPSS FD001

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange?logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Predict the **Remaining Useful Life (RUL)** of industrial turbine engines before failure occurs — comparing LSTM (PyTorch) vs XGBoost on the NASA CMAPSS FD001 benchmark dataset, with a two-tab industrial dashboard deployed on Streamlit Cloud.

---

## Business Case

Unplanned turbofan engine failures cost the aviation and industrial sector an estimated **$500K+ per event** in emergency repairs, unscheduled downtime, and safety incidents. Early prediction turns reactive maintenance into planned maintenance.

| KPI | Value |
|-----|-------|
| Cost avoided per failure | ~$450K (replace emergency repair with planned maintenance) |
| Production hours recovered | Configurable in dashboard based on fleet size and hourly rate |

---

## Live Demo

**[🚀 Open Dashboard →](https://predictive-maintenance-rul-sarsrm36mxc5zmbmnn6hky.streamlit.app/)**
<!-- Add animated GIF of dashboard here -->

---

## Results

| Model       | RMSE      | MAE   | NASA Score |
|-------------|-----------|-------|------------|
| **LSTM** ✅ | **17.08** | **12.50** | **493**  |
| XGBoost     | 86.20     | 75.52 | 559,018    |

> LSTM wins decisively — sequence modeling captures temporal degradation patterns that tabular XGBoost misses.

---

## Architecture

```
Raw Sensors → Feature Engineering → LSTM / XGBoost → RUL Pipeline → Streamlit Dashboard
               (rolling stats,        (MC Dropout CI /    (unified API)   (2 tabs: Ops + Analytics)
                MinMax scaler)         Bootstrap CI)
```

**Key design principle:** `src/` is a standalone Python package. Notebooks consume it; the dashboard imports it. Changing a model doesn't break the dashboard.

---

## Dataset

**NASA CMAPSS FD001** — 100 simulated turbofan engines, run to failure.
- 21 sensor measurements per cycle
- 3 operational settings
- ~20,600 training rows across 100 engines

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
cd <YOUR_REPO>

# GPU training (RTX 3070/4070/4090 — CUDA 12.1):
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Then install remaining dependencies:
pip install -r requirements.txt
```

### 2. Download data

```bash
# Requires ~/.kaggle/kaggle.json (free Kaggle account)
python scripts/download_data.py
```

### 3. Run notebooks in order

```bash
jupyter notebook notebooks/
```

| Notebook | Purpose |
|----------|---------|
| `01_EDA.ipynb` | Sensor variance analysis, degradation patterns |
| `02_Preprocessing.ipynb` | Feature engineering, scaler fitting, parquet export |
| `03_Modeling.ipynb` | Train LSTM + XGBoost, save model files |
| `04_Evaluation.ipynb` | RMSE, MAE, NASA Score, prediction scatter plots |

### 4. Launch dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
├── data/
│   ├── raw/                    # CMAPSS files (gitignored)
│   └── processed/              # Engineered features as Parquet (gitignored)
├── notebooks/                  # 01 EDA → 02 Preprocessing → 03 Modeling → 04 Evaluation
├── src/
│   ├── data/loader.py          # Load raw files, compute RUL, split by engine
│   ├── features/engineering.py # Rolling stats, sensor filtering, MinMax scaler
│   ├── models/lstm_model.py    # PyTorch LSTM + MC Dropout uncertainty
│   ├── models/xgboost_model.py # XGBoost + bootstrap confidence intervals
│   └── inference/pipeline.py  # Unified RULPipeline (load + predict)
├── app/
│   ├── streamlit_app.py        # Entry point
│   ├── tabs/operations.py      # Tab 1: Technician view (fleet table, sensor charts)
│   └── tabs/analytics.py       # Tab 2: Manager view (KPIs, business case)
├── scripts/
│   └── download_data.py        # Kaggle API download
├── models/                     # Saved weights (.pt, .pkl) — gitignored locally
└── .github/workflows/tests.yml # CI: flake8 + pytest on push
```

---

## Features

**Tab 1 — Operaciones (Maintenance Technician)**
- Fleet status table with traffic-light indicators (🔴 Critical / 🟡 Warning / 🟢 Normal)
- Per-engine sensor degradation charts
- RUL prediction with 95% confidence interval (MC Dropout for LSTM, bootstrap for XGBoost)
- One-click CSV fleet report export

**Tab 2 — Analytics (Plant Manager)**
- KPI dashboard: critical engines, avg fleet RUL, estimated cost avoided
- Model comparison table (LSTM vs XGBoost)
- RUL distribution histogram across fleet
- Business case calculator with configurable costs

---

## Model Details

| | LSTM | XGBoost |
|-|------|---------|
| Input | 30-cycle sliding window | Last-window feature vector |
| Architecture | 2-layer LSTM → Linear(64→32→1) | n_estimators=500, depth=6 |
| Uncertainty | Monte Carlo Dropout (50 passes) | Bootstrap ensemble (20 models) |
| Output | RUL + 95% CI | RUL + 5th/95th percentile CI |

---

## Built With

- **PyTorch** — LSTM with MC Dropout
- **XGBoost** — Gradient boosting with bootstrap CI
- **scikit-learn** — MinMaxScaler
- **Streamlit** — Interactive dashboard
- **Plotly** — Charts and visualizations
- **pandas / numpy** — Data processing
- **Kaggle API** — Dataset download

---

## License

MIT — feel free to use this for your own portfolio.

---

## Contact

Built by **Favian Zamorano** · [LinkedIn](https://linkedin.com/in/favian-zamorano) · [Portfolio](https://github.com/FavianZamorano)
