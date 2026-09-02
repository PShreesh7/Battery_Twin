import plotly.graph_objects as go
import numpy as np
import pandas as pd

def create_capacity_chart(df: pd.DataFrame, cell_id: str, init_cap: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["cycle"],
        y=df["capacity"],
        mode="lines+markers",
        name=f"Cell {cell_id} Capacity",
        line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=4)
    ))
    fig.add_hline(
        y=init_cap * 0.80,
        line_dash="dash",
        line_color="red",
        annotation_text="80% EOL Threshold",
        annotation_position="bottom right"
    )
    fig.update_layout(
        title=f"Capacity Degradation History - {cell_id}",
        xaxis_title="Cycle Number",
        yaxis_title="Discharge Capacity (Ah)",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

def create_telemetry_chart(df: pd.DataFrame, cell_id: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["voltage_mean"],
        mode="lines", name="Mean Voltage (V)",
        line=dict(color="#2ca02c", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["temp_mean"],
        mode="lines", name="Mean Temperature (?C)",
        line=dict(color="#d62728", width=2),
        yaxis="y2"
    ))
    fig.update_layout(
        title=f"Multi-Signal Operating Telemetry - {cell_id}",
        xaxis_title="Cycle Number",
        yaxis=dict(title="Voltage (V)", titlefont=dict(color="#2ca02c")),
        yaxis2=dict(
            title="Temperature (?C)",
            titlefont=dict(color="#d62728"),
            overlaying="y",
            side="right"
        ),
        template="plotly_white",
        legend=dict(x=0.01, y=0.99)
    )
    return fig

def create_forecast_chart(
    hist_df: pd.DataFrame,
    forecast_cycles: list,
    forecast_caps: list,
    model_name: str,
    lower_bound: list = None,
    upper_bound: list = None,
    init_cap: float = 2.0
) -> go.Figure:
    fig = go.Figure()
    # Historical
    fig.add_trace(go.Scatter(
        x=hist_df["cycle"],
        y=hist_df["capacity"],
        mode="lines",
        name="Historical Measured",
        line=dict(color="black", width=2)
    ))
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_cycles,
        y=forecast_caps,
        mode="lines+markers",
        name=f"Predicted ({model_name})",
        line=dict(color="#00cc96", width=2.5)
    ))
    # Uncertainty bounds
    if lower_bound is not None and upper_bound is not None:
        fig.add_trace(go.Scatter(
            x=forecast_cycles + forecast_cycles[::-1],
            y=upper_bound + lower_bound[::-1],
            fill="toself",
            fillcolor="rgba(0, 204, 150, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="90% Confidence Interval"
        ))
    fig.add_hline(
        y=init_cap * 0.80,
        line_dash="dash",
        line_color="red",
        annotation_text="80% EOL Threshold"
    )
    fig.update_layout(
        title=f"Battery Digital Twin Prognostic Forecast ({model_name})",
        xaxis_title="Cycle Number",
        yaxis_title="Capacity (Ah)",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig
