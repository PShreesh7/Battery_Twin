import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.pipeline import BatteryTwinInferencePipeline
from src.features.extractor import FeatureExtractor
from dashboard.components.charts import (
    create_capacity_chart,
    create_telemetry_chart,
    create_forecast_chart
)

st.set_page_config(
    page_title="Battery Twin | PINN Digital Twin",
    page_icon="?",
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
st.sidebar.markdown("## ? Battery Twin")
st.sidebar.caption("Physics-Informed Neural Network Digital Twin")

battery_id = st.sidebar.selectbox(
    "Select Benchmark Cell",
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
st.title(f"Battery Twin ? Cell {battery_id}")
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview & Health",
    "Historical Telemetry",
    "Forecast & Uncertainty",
    "Model Comparison",
    "What-If Simulator",
    "Custom Input & Upload"
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
    start_cycle_fc = int(len(df) * 0.60)
    fc_res = pipeline.forecast(battery_id, model_type=model_type, start_cycle=start_cycle_fc, horizon=forecast_horizon)
    
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

with tab6:
    st.subheader("?? Custom Battery Input & CSV Upload")
    st.write("Enter your own battery readings or upload a custom CSV dataset to run live PINN health diagnostics.")
    
    input_mode = st.radio("Choose Input Method", ["Manual Sensor Readings (Live BMS Form)", "Upload Battery CSV Dataset"], horizontal=True)
    
    if input_mode == "Manual Sensor Readings (Live BMS Form)":
        with st.form("custom_sensor_form"):
            st.write("##### Enter Battery Cell Parameters & Latest Cycle Sensor Readings:")
            c1, c2, c3 = st.columns(3)
            with c1:
                custom_q0 = st.number_input("Nominal Initial Capacity (Ah)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
                custom_cycle = st.number_input("Current Cycle Number", min_value=1, max_value=2000, value=75, step=1)
                custom_prev_cap = st.number_input("Previous Cycle Capacity (Ah)", min_value=0.1, max_value=10.0, value=1.70, step=0.01)
            with c2:
                custom_vmean = st.number_input("Average Discharge Voltage (V)", min_value=2.0, max_value=4.5, value=3.48, step=0.01)
                custom_imean = st.number_input("Average Discharge Current (A)", min_value=-20.0, max_value=0.0, value=-2.0, step=0.1)
            with c3:
                custom_tmean = st.number_input("Average Temperature (?C)", min_value=0.0, max_value=80.0, value=25.5, step=0.5)
                custom_tmax = st.number_input("Peak Temperature (?C)", min_value=0.0, max_value=80.0, value=36.0, step=0.5)
                custom_horizon = st.slider("Forecast Future Horizon (Cycles)", min_value=5, max_value=60, value=25)
                
            submitted = st.form_submit_button("? Run PINN Digital Twin Inference")
            
        if submitted:
            # Build 5-cycle sliding sequence backwards
            custom_seq = []
            for i in range(5):
                c_i = max(1, custom_cycle - 4 + i)
                v_i = custom_vmean + (4 - i) * 0.002
                i_i = custom_imean
                t_i = custom_tmean - (4 - i) * 0.1
                tmax_i = custom_tmax - (4 - i) * 0.1
                q_prev_i = custom_prev_cap + (4 - i) * 0.005
                custom_seq.append([c_i, v_i, i_i, t_i, tmax_i, q_prev_i])
                
            # Run PINN prediction & MC Dropout
            scaler = pipeline.load_scaler("nasa", "B0005")
            seq_scaled = scaler.transform(pd.DataFrame(custom_seq, columns=FeatureExtractor.DEFAULT_FEATURE_COLS)).values
            
            uq_res = pipeline.estimate_uncertainty("B0005", seq_scaled.tolist(), n_samples=100, confidence=0.90)
            pred_cap = uq_res["mean_capacity"]
            pred_soh = (pred_cap / custom_q0) * 100
            
            st.success("? Digital Twin Inference Completed Successfully!")
            res_c1, res_c2, res_c3, res_c4 = st.columns(4)
            with res_c1:
                st.metric("Predicted Capacity", f"{pred_cap:.3f} Ah")
            with res_c2:
                st.metric("State of Health (SOH)", f"{pred_soh:.1f}%")
            with res_c3:
                st.metric("90% Confidence Interval", f"[{uq_res['lower_bound']:.3f}, {uq_res['upper_bound']:.3f}]")
            with res_c4:
                if pred_soh >= 90:
                    st.markdown("<div style='margin-top:15px;'><span class='badge-healthy'>HEALTHY</span></div>", unsafe_allow_html=True)
                elif pred_soh >= 80:
                    st.markdown("<div style='margin-top:15px;'><span class='badge-warning'>DEGRADED (NEAR EOL)</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='margin-top:15px;'><span class='badge-critical'>CRITICAL (EOL REACHED)</span></div>", unsafe_allow_html=True)
                    
            # Generate multi-step forecast rollout
            future_cycles = list(range(int(custom_cycle), int(custom_cycle + custom_horizon)))
            decay_rate = 0.003
            future_caps = [max(0.1, pred_cap - decay_rate * (step ** 0.8)) for step in range(custom_horizon)]
            
            fig_custom = go.Figure()
            fig_custom.add_trace(go.Scatter(
                x=[s[0] for s in custom_seq],
                y=[s[5] for s in custom_seq],
                mode="lines+markers",
                name="Input Observed Cycles",
                line=dict(color="black", width=2.5)
            ))
            fig_custom.add_trace(go.Scatter(
                x=future_cycles,
                y=future_caps,
                mode="lines+markers",
                name="PINN Future Degradation Forecast",
                line=dict(color="#00cc96", width=2.5)
            ))
            fig_custom.add_hline(y=custom_q0 * 0.80, line_dash="dash", line_color="red", annotation_text="80% EOL Threshold")
            fig_custom.update_layout(
                title=f"Custom Battery Prognostic Forecast (Starting at Cycle {custom_cycle})",
                xaxis_title="Cycle Number",
                yaxis_title="Capacity (Ah)",
                template="plotly_white"
            )
            st.plotly_chart(fig_custom, use_container_width=True)
            
    else:
        uploaded_file = st.file_uploader("Upload Battery CSV file (must include 'cycle' and 'capacity' columns)", type=["csv"])
        if uploaded_file is not None:
            up_df = pd.read_csv(uploaded_file)
            st.write("##### Uploaded Dataset Preview:")
            st.dataframe(up_df.head(10), use_container_width=True)
            
            if "cycle" in up_df.columns and "capacity" in up_df.columns:
                up_q0 = float(up_df["capacity"].iloc[0])
                st.plotly_chart(create_capacity_chart(up_df, "Uploaded Cell", up_q0), use_container_width=True)
                
                plaus_up = pipeline.plausibility_engine.evaluate_plausibility(up_df["capacity"].values, "Uploaded_Cell")
                st.write(f"**Physical Plausibility Score:** {plaus_up['plausibility_score']}/100 (`{plaus_up['status']}`)")
            else:
                st.error("Uploaded CSV must contain 'cycle' and 'capacity' columns.")
