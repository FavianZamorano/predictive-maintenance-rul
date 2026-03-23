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
        raw_df:  original sensor dataframe for degradation charts
    """
    st.header("Fleet Status")

    # Fleet table
    display = results.copy()
    display["Status"] = display["status"].map(
        lambda s: f"{STATUS_EMOJI[s]} {STATUS_LABEL[s]}"
    )
    display = display.rename(columns={
        "engine_id": "Engine",
        "rul_pred":  "RUL (cycles)",
        "ci_lower":  "CI Lower",
        "ci_upper":  "CI Upper",
    })
    st.dataframe(
        display[["Engine", "RUL (cycles)", "CI Lower", "CI Upper", "Status"]],
        use_container_width=True,
        hide_index=True,
    )

    # Engine selector
    engine_id = st.selectbox(
        "Select engine for detail view",
        options=sorted(results["engine_id"].unique()),
    )
    engine_data   = raw_df[raw_df["engine_id"] == engine_id].sort_values("cycle")
    engine_result = results[results["engine_id"] == engine_id].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted RUL", f"{engine_result['rul_pred']:.0f} cycles")
    with col2:
        status = engine_result["status"]
        st.markdown(
            f"**Status:** <span class='status-{status}'>"
            f"{STATUS_EMOJI[status]} {status.upper()}</span>",
            unsafe_allow_html=True,
        )
    st.caption(
        f"95% Confidence Interval: "
        f"[{engine_result['ci_lower']:.0f}, {engine_result['ci_upper']:.0f}] cycles"
    )

    # Sensor degradation chart
    st.subheader("Sensor Degradation")
    sensor_cols = [c for c in engine_data.columns if c.startswith("sensor_")]
    if sensor_cols:
        selected_sensor = st.selectbox("Select sensor", sensor_cols[:12])
        fig = px.line(
            engine_data, x="cycle", y=selected_sensor,
            title=f"Engine {engine_id} — {selected_sensor}",
            labels={"cycle": "Cycle", selected_sensor: "Value"},
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    # CSV export
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export Fleet Report (CSV)",
        csv,
        file_name="fleet_rul_report.csv",
        mime="text/csv",
    )
