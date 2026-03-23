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
    critical = results[results["status"] == "critical"]
    warning  = results[results["status"] == "warning"]
    avg_rul  = results["rul_pred"].mean()

    engines_at_risk   = len(critical)
    savings_per_engine = cost_failure - cost_preventive
    total_savings      = engines_at_risk * savings_per_engine
    mean_lead_time     = critical["rul_pred"].mean() if engines_at_risk > 0 else 0.0
    downtime_hours     = engines_at_risk * mean_lead_time * 1.0
    downtime_value     = downtime_hours * prod_value_per_hour

    # KPI cards
    st.header("Fleet Analytics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔴 Critical Engines", engines_at_risk)
    col2.metric("🟡 Warning Engines",  len(warning))
    col3.metric("Avg Fleet RUL",       f"{avg_rul:.0f} cycles")
    col4.metric("Est. Cost Avoided",   f"${total_savings:,.0f}")

    st.divider()

    # Model comparison
    st.subheader("Model Performance (Official Test Set)")
    try:
        metrics = pd.read_csv("docs/metrics_summary.csv")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.info("Run notebook 04_Evaluation.ipynb to populate metrics.")

    # RUL distribution histogram
    st.subheader("Fleet RUL Distribution")
    fig = px.histogram(
        results, x="rul_pred", nbins=20,
        color="status",
        color_discrete_map={
            "critical": "#ff4b4b",
            "warning":  "#ffa500",
            "normal":   "#00d4aa",
        },
        template="plotly_dark",
        labels={"rul_pred": "Predicted RUL (cycles)"},
        title="Distribution of Remaining Useful Life Across Fleet",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Business case
    st.subheader("Business Case Calculator")
    st.caption("Assumption: 1 cycle ≈ 1 hour of operation")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cost Savings from Early Detection**")
        fig_bar = go.Figure(go.Bar(
            x=["Unplanned Failure", "Preventive Maint.", "Net Savings"],
            y=[
                cost_failure    * max(engines_at_risk, 1),
                cost_preventive * max(engines_at_risk, 1),
                total_savings,
            ],
            marker_color=["#ff4b4b", "#ffa500", "#00d4aa"],
        ))
        fig_bar.update_layout(template="plotly_dark", yaxis_title="Cost ($)")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("**Production Value Recovered**")
        st.metric("Downtime hours avoided", f"{downtime_hours:.0f} hrs")
        st.metric("Production value recovered", f"${downtime_value:,.0f}")
        st.caption(
            f"{engines_at_risk} critical engines × "
            f"{mean_lead_time:.0f} cycles × "
            f"${prod_value_per_hour:,.0f}/hr"
        )
