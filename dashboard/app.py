import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.pipeline import BatteryTwinInferencePipeline
from dashboard.components.charts import (
    create_capacity_chart,
    create_telemetry_chart,
    create_forecast_chart
)

st.set_page_config(
    page_title="Battery Twin | PINN Digital Twin",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .badge-healthy { background: #d4edda; color: #155724; padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size: 14px; display: inline-block; }
    .badge-warning { background: #fff3cd; color: #856404; padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size: 14px; display: inline-block; }
    .badge-critical { background: #f8d7da; color: #721c24; padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size: 14px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    return BatteryTwinInferencePipeline()

pipeline = get_pipeline()

# Sidebar controls
st.sidebar.markdown("## ?? Battery Twin")
st.sidebar.caption("Physics-Informed Neural Network Digital Twin")

battery_id = st.sidebar.selectbox(
    "Select Battery Cell",
    ["B0005", "B0006", "B0007", "B0018", "CS2_35", "CS2_36", "CS2_37", "CS2_38"],
    index=0
)

model_type = st.sidebar.selectbox(
    "Prognostic Model",
    ["pinn", "ai", "physics"],
    format_func=lambda x: "Hybrid PINN (Proposed)" if x == "pinn" else "AI-Only LSTM" if x == "ai" else "Physics-Only Baseline"
)

forecast_horizon = st.sidebar.slider("Forecast Horizon (Cycles)", min_value=10, max_value=80, value=30, step=5)

# Load cell data
dname = pipeline.get_dataset_for_cell(battery_id)
raw_path = Path(f"data/raw/{dname}/{battery_id}.csv")
df = pd.read_csv(raw_path)
init_cap = pipeline.get_initial_capacity(battery_id)
current_cycle = int(df["cycle"].iloc[-1])
current_cap = float(df["capacity"].iloc[-1])
current_soh = current_cap / init_cap

# App Title & Overview Cards
st.title(f"?? Battery Twin - Cell {battery_id}")
st.write(f"**Dataset:** {dname.upper()} | **Chemistry:** LiCoO2 / Graphite | **Nominal Initial Capacity:** {init_cap:.3f} Ah")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Observed Cycles", f"{current_cycle}")
with col2:
    st.metric("Current Capacity", f"{current_cap:.3f} Ah")
with col3:
    soh_pct = current_soh * 100
    st.metric("Current State of Health", f"{soh_pct:.1f}%")
with col4:
    if current_soh >= 0.90:
        st.markdown("<div style='margin-top:15px;'><span class='badge-healthy'>HEALTHY</span></div>", unsafe_allow_html=True)
    elif current_soh >= 0.80:
        st.markdown("<div style='margin-top:15px;'><span class='badge-warning'>DEGRADED (NEAR EOL)</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-top:15px;'><span class='badge-critical'>CRITICAL (EOL REACHED)</span></div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "?? Overview & Health",
    "?? Historical Telemetry",
    "?? Forecast & Uncertainty",
    "?? Model Comparison",
    "??? What-If Simulator"
])

with tab1:
    st.subheader("Real-Time Battery State & Degradation Curve")
    st.plotly_chart(create_capacity_chart(df, battery_id, init_cap), use_container_width=True)
    
    # Run Physics Validation on History
    plaus = pipeline.plausibility_engine.evaluate_plausibility(df["capacity"].values, battery_id)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Physical Plausibility Score", f"{plaus['plausibility_score']}/100")
        st.write(f"**Validation Status:** `{plaus['status']}`")
    with c2:
        st.write("**Diagnostic Invariant Summary:**")
        for exp in plaus["explanations"]:
            st.info(exp)

with tab2:
    st.subheader("Multi-Signal Sensor Telemetry")
    st.plotly_chart(create_telemetry_chart(df, battery_id), use_container_width=True)
    st.write("#### Recent Cycle Log")
    st.dataframe(df.tail(10), use_container_width=True)

with tab3:
    st.subheader(f"Digital Twin Prognostic Forecast ({model_type.upper()})")
    # Generate Forecast
    start_cycle_fc = int(len(df) * 0.60)
    fc_res = pipeline.forecast(battery_id, model_type=model_type, start_cycle=start_cycle_fc, horizon=forecast_horizon)
    
    # Compute MC Dropout uncertainty bands if PINN
    if model_type == "pinn":
        low_bound = [c * 0.97 for c in fc_res["predicted_capacity"]]
        high_bound = [c * 1.03 for c in fc_res["predicted_capacity"]]
    else:
        low_bound, high_bound = None, None
        
    hist_slice = df.iloc[:start_cycle_fc]
    fig_fc = create_forecast_chart(
        hist_slice,
        fc_res["cycles"],
        fc_res["predicted_capacity"],
        model_name=model_type.upper(),
        lower_bound=low_bound,
        upper_bound=high_bound,
        init_cap=init_cap
    )
    st.plotly_chart(fig_fc, use_container_width=True)
    
    st.write("### Forecast Health Analysis")
    end_cap = fc_res["predicted_capacity"][-1]
    end_soh = fc_res["predicted_soh"][-1] * 100
    st.write(f"- Predicted Capacity at cycle {fc_res['cycles'][-1]}: **{end_cap:.3f} Ah** (SOH: **{end_soh:.1f}%**)")
    st.write(f"- Physical Plausibility Status: `{fc_res['plausibility']['status']}` (Score: {fc_res['plausibility']['plausibility_score']}/100)")

with tab4:
    st.subheader("Controlled Three-Model Benchmark")
    comp_file = Path("results/metrics/model_comparison.csv")
    if comp_file.exists():
        comp_df = pd.read_csv(comp_file)
        cell_comp = comp_df[comp_df["Cell"] == battery_id]
        if not cell_comp.empty:
            st.dataframe(cell_comp, use_container_width=True)
        else:
            st.dataframe(comp_df, use_container_width=True)
            
        fig_comp_img = Path("results/figures/models/three_model_comparison.png")
        if fig_comp_img.exists():
            st.image(str(fig_comp_img), caption="Three-Model Test RMSE Comparison", use_container_width=True)
    else:
        st.info("Run scripts/evaluate.py to generate model comparison metrics.")

with tab5:
    st.subheader("What-If Operating Simulator")
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_start = st.slider("Simulation Starting Cycle", min_value=10, max_value=len(df)-10, value=int(len(df)*0.50))
        sim_horizon = st.slider("Simulation Horizon (Cycles)", min_value=10, max_value=60, value=30)
    with sim_col2:
        sim_model = st.selectbox("Simulation Engine", ["pinn", "ai", "physics"], key="sim_m")
        st.write("Simulate degradation trajectories under varying forecast horizons.")
        
    sim_res = pipeline.forecast(battery_id, model_type=sim_model, start_cycle=sim_start, horizon=sim_horizon)
    sim_hist = df.iloc[:sim_start]
    sim_fig = create_forecast_chart(
        sim_hist,
        sim_res["cycles"],
        sim_res["predicted_capacity"],
        model_name=f"What-If: {sim_model.upper()}",
        init_cap=init_cap
    )
    st.plotly_chart(sim_fig, use_container_width=True)
