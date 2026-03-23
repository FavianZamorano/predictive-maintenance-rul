"""Predictive Maintenance Dashboard — entry point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from src.data.loader import load_cmapss_raw, COLUMNS
from src.inference.pipeline import RULPipeline
from app.tabs import operations, analytics

st.set_page_config(
    page_title="Industrial RUL Predictor",
    page_icon="🔧",
    layout="wide",
)

# Custom CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔧 RUL Predictor")
    st.caption("NASA CMAPSS FD001 · Turbine Engine Fleet")
    st.divider()

    uploaded = st.file_uploader("Upload sensor CSV", type=["csv", "txt"])
    model_choice = st.selectbox("Model", ["LSTM", "XGBoost"])

    st.subheader("Cost Parameters")
    cost_failure    = st.number_input("Unplanned failure ($)",      value=500_000, step=10_000)
    cost_preventive = st.number_input("Preventive maintenance ($)", value=50_000,  step=1_000)
    prod_value      = st.number_input("Production value ($/hr)",    value=5_000,   step=500)


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_demo():
    _, test_raw, _ = load_cmapss_raw(data_dir="data/raw")
    return test_raw


if uploaded is not None:
    raw_df = pd.read_csv(uploaded, sep=r"\s+", header=None, engine="python")
    raw_df = raw_df.iloc[:, :len(COLUMNS)]
    raw_df.columns = COLUMNS
else:
    try:
        raw_df = load_demo()
    except FileNotFoundError:
        st.error(
            "Demo data not found. Run `python scripts/download_data.py` first, "
            "or upload a CSV file in the sidebar."
        )
        st.stop()


# ── Inference ─────────────────────────────────────────────────────────────────
@st.cache_data
def run_inference(model_type: str, _df: pd.DataFrame):
    try:
        pipeline = RULPipeline(model_dir="models", model_type=model_type)
        return pipeline.predict(_df)
    except FileNotFoundError as e:
        return None, str(e)


model_type = "lstm" if model_choice == "LSTM" else "xgboost"
result = run_inference(model_type, raw_df)

if isinstance(result, tuple):
    st.error(f"Model files not found: {result[1]}. Run notebooks 02 and 03 first.")
    st.stop()

results = result

# ── Dashboard ─────────────────────────────────────────────────────────────────
st.title("🔧 Industrial RUL Predictor")
st.caption("Remaining Useful Life prediction for turbine engine fleets — NASA CMAPSS FD001")

tab1, tab2 = st.tabs(["⚙️ Operaciones", "📊 Analytics"])

with tab1:
    operations.render(results, raw_df)

with tab2:
    analytics.render(results, cost_failure, cost_preventive, prod_value)
