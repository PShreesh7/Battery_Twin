import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# Design System Palette: Industrial Intelligence Console
COLOR_INK = "#172B3A"
COLOR_SLATE = "#607180"
COLOR_GRID = "#E5EBEF"
COLOR_BORDER = "#DCE4E9"
COLOR_BG = "#FFFFFF"

COLOR_ACCENT_TEAL = "#087F8C"      # Primary Accent: Hybrid PINN / Capacity
COLOR_ACCENT_SAGE = "#39A6A3"      # Secondary Accent
COLOR_WARNING = "#B7791F"          # Warning / Thermal profile
COLOR_CRITICAL = "#C63D4F"         # Critical / 80% EOL threshold
COLOR_AI_BASELINE = "#D97706"      # AI-only LSTM (Amber)
COLOR_PHYSICS_BASELINE = "#64748B" # Physics-only baseline (Slate)

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
MONO_FONT = "'JetBrains Mono', Consolas, monospace"


def apply_industrial_layout(fig: go.Figure, title_text: str, x_title: str, y_title: str) -> go.Figure:
    """Applies standardized Industrial Intelligence Console layout to a Plotly figure."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title_text}</b>",
            font=dict(family=FONT_FAMILY, size=15, color=COLOR_INK),
            x=0.01,
            xanchor="left",
            y=0.96
        ),
        margin=dict(l=55, r=40, t=55, b=45),
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        font=dict(family=FONT_FAMILY, color=COLOR_SLATE, size=12),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=COLOR_INK,
            font_size=12,
            font_family=MONO_FONT,
            font_color="#FFFFFF"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(family=FONT_FAMILY, size=11, color=COLOR_INK),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=COLOR_BORDER,
            borderwidth=1
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(family=FONT_FAMILY, size=12, color=COLOR_INK)),
            gridcolor=COLOR_GRID,
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=COLOR_BORDER,
            linewidth=1,
            tickfont=dict(family=MONO_FONT, size=11, color=COLOR_SLATE)
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(family=FONT_FAMILY, size=12, color=COLOR_INK)),
            gridcolor=COLOR_GRID,
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=COLOR_BORDER,
            linewidth=1,
            tickfont=dict(family=MONO_FONT, size=11, color=COLOR_SLATE)
        )
    )
    return fig


def create_capacity_chart(df: pd.DataFrame, cell_id: str, init_cap: float) -> go.Figure:
    """Creates a degradation curve showing historical capacity and 80% EOL threshold."""
    fig = go.Figure()
    
    # Capacity degradation trace
    fig.add_trace(go.Scatter(
        x=df["cycle"],
        y=df["capacity"],
        mode="lines+markers",
        name=f"Cell {cell_id} Capacity",
        line=dict(color=COLOR_ACCENT_TEAL, width=2.5),
        marker=dict(size=4, color=COLOR_ACCENT_TEAL),
        hovertemplate="Cycle %{x}<br>Capacity: <b>%{y:.4f} Ah</b><extra></extra>"
    ))
    
    # 80% EOL threshold line
    eol_val = init_cap * 0.80
    fig.add_hline(
        y=eol_val,
        line_dash="dash",
        line_color=COLOR_CRITICAL,
        line_width=1.5,
        annotation_text=f"80% EOL Threshold ({eol_val:.3f} Ah)",
        annotation_position="bottom right",
        annotation_font=dict(family=MONO_FONT, size=11, color=COLOR_CRITICAL),
        annotation_bgcolor="rgba(253, 236, 238, 0.8)"
    )
    
    apply_industrial_layout(
        fig,
        title_text=f"Capacity Degradation History — Cell {cell_id}",
        x_title="Discharge Cycle",
        y_title="Discharge Capacity (Ah)"
    )
    return fig


def create_telemetry_chart(df: pd.DataFrame, cell_id: str) -> go.Figure:
    """Creates synchronized dual-axis sensor telemetry chart for voltage and temperature."""
    fig = go.Figure()
    
    # Voltage trace (Primary axis)
    fig.add_trace(go.Scatter(
        x=df["cycle"],
        y=df["voltage_mean"],
        mode="lines",
        name="Mean Voltage (V)",
        line=dict(color=COLOR_ACCENT_TEAL, width=2.2),
        hovertemplate="Voltage: <b>%{y:.3f} V</b><extra></extra>"
    ))
    
    # Temperature trace (Secondary axis)
    fig.add_trace(go.Scatter(
        x=df["cycle"],
        y=df["temp_mean"],
        mode="lines",
        name="Mean Temperature (°C)",
        line=dict(color=COLOR_WARNING, width=2.2),
        yaxis="y2",
        hovertemplate="Temp: <b>%{y:.1f} °C</b><extra></extra>"
    ))
    
    apply_industrial_layout(
        fig,
        title_text=f"Multi-Signal Operating Telemetry — Cell {cell_id}",
        x_title="Cycle Number",
        y_title="Mean Voltage (V)"
    )
    
    # Configure secondary y-axis for temperature
    fig.update_layout(
        yaxis=dict(
            title=dict(text="Discharge Voltage (V)", font=dict(family=FONT_FAMILY, size=12, color=COLOR_ACCENT_TEAL)),
            tickfont=dict(family=MONO_FONT, color=COLOR_ACCENT_TEAL, size=11),
            gridcolor=COLOR_GRID,
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=COLOR_BORDER
        ),
        yaxis2=dict(
            title=dict(text="Operating Temp (°C)", font=dict(family=FONT_FAMILY, size=12, color=COLOR_WARNING)),
            tickfont=dict(family=MONO_FONT, color=COLOR_WARNING, size=11),
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor=COLOR_BORDER
        )
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
    """Creates a prognostic trajectory chart with historical telemetry and forecast uncertainty."""
    fig = go.Figure()
    
    # Model color selection
    model_lower = model_name.lower()
    if "pinn" in model_lower:
        forecast_color = COLOR_ACCENT_TEAL
    elif "ai" in model_lower:
        forecast_color = COLOR_AI_BASELINE
    else:
        forecast_color = COLOR_PHYSICS_BASELINE

    # Historical measured trace
    fig.add_trace(go.Scatter(
        x=hist_df["cycle"],
        y=hist_df["capacity"],
        mode="lines",
        name="Historical Measured",
        line=dict(color=COLOR_INK, width=2),
        hovertemplate="Observed Cycle %{x}: <b>%{y:.4f} Ah</b><extra></extra>"
    ))
    
    # Uncertainty envelope (90% Confidence Interval)
    if lower_bound is not None and upper_bound is not None:
        fig.add_trace(go.Scatter(
            x=list(forecast_cycles) + list(forecast_cycles)[::-1],
            y=list(upper_bound) + list(lower_bound)[::-1],
            fill="toself",
            fillcolor="rgba(8, 127, 140, 0.15)",
            line=dict(color="rgba(8, 127, 140, 0.3)", width=1),
            hoverinfo="skip",
            showlegend=True,
            name="90% Confidence Interval (MC Dropout)"
        ))
    
    # Forecast predicted trace
    fig.add_trace(go.Scatter(
        x=forecast_cycles,
        y=forecast_caps,
        mode="lines+markers",
        name=f"Forecast ({model_name})",
        line=dict(color=forecast_color, width=2.5),
        marker=dict(size=4, color=forecast_color),
        hovertemplate="Forecast Cycle %{x}: <b>%{y:.4f} Ah</b><extra></extra>"
    ))
    
    # 80% EOL threshold line
    eol_val = init_cap * 0.80
    fig.add_hline(
        y=eol_val,
        line_dash="dash",
        line_color=COLOR_CRITICAL,
        line_width=1.5,
        annotation_text=f"80% EOL Threshold ({eol_val:.3f} Ah)",
        annotation_position="bottom right",
        annotation_font=dict(family=MONO_FONT, size=11, color=COLOR_CRITICAL),
        annotation_bgcolor="rgba(253, 236, 238, 0.8)"
    )
    
    apply_industrial_layout(
        fig,
        title_text=f"Battery Digital Twin Prognostic Forecast — {model_name}",
        x_title="Discharge Cycle",
        y_title="Capacity (Ah)"
    )
    return fig
