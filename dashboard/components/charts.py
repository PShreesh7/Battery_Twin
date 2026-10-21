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


def create_overview_degradation_chart(
    hist_df: pd.DataFrame,
    forecast_cycles: list,
    forecast_caps: list,
    lower_bound: list = None,
    upper_bound: list = None,
    init_cap: float = 2.0,
    cell_id: str = "B0005",
    model_name: str = "PINN"
) -> go.Figure:
    """
    Creates an all-in-one capacity-vs-cycle chart for Overview Mode:
    Includes historical measurements, forecast trajectory, forecast boundary,
    EOL threshold, and optional uncertainty bands.
    """
    fig = go.Figure()
    
    # 1. Historical capacity
    fig.add_trace(go.Scatter(
        x=hist_df["cycle"],
        y=hist_df["capacity"],
        mode="lines",
        name="Historical Measured",
        line=dict(color=COLOR_INK, width=2.2),
        hovertemplate="Observed Cycle %{x}: <b>%{y:.4f} Ah</b><extra></extra>"
    ))
    
    # 2. 90% Confidence Interval envelope (if PINN / provided)
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
        
    # 3. Forecast trajectory
    model_lower = model_name.lower()
    color_map = {
        "pinn": COLOR_ACCENT_TEAL,
        "ai": COLOR_AI_BASELINE,
        "physics": COLOR_PHYSICS_BASELINE
    }
    f_color = color_map.get(model_lower, COLOR_ACCENT_TEAL)
    
    fig.add_trace(go.Scatter(
        x=forecast_cycles,
        y=forecast_caps,
        mode="lines+markers",
        name=f"Forecast ({model_name.upper()})",
        line=dict(color=f_color, width=2.6),
        marker=dict(size=4, color=f_color),
        hovertemplate="Forecast Cycle %{x}: <b>%{y:.4f} Ah</b><extra></extra>"
    ))
    
    # 4. Vertical Forecast Boundary Line
    if len(forecast_cycles) > 0:
        split_c = forecast_cycles[0]
        fig.add_vline(
            x=split_c,
            line_dash="dot",
            line_color=COLOR_SLATE,
            line_width=1.5,
            annotation_text="Forecast Boundary",
            annotation_position="top left",
            annotation_font=dict(family=MONO_FONT, size=10, color=COLOR_SLATE)
        )
    
    # 5. 80% EOL Threshold line
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
        title_text=f"Battery Capacity Degradation & Digital Twin Forecast — Cell {cell_id}",
        x_title="Discharge Cycle",
        y_title="Discharge Capacity (Ah)"
    )
    return fig


def create_telemetry_chart(
    df: pd.DataFrame,
    cell_id: str,
    cycle_min: int = None,
    cycle_max: int = None
) -> go.Figure:
    """Creates synchronized dual-axis sensor telemetry chart for voltage and temperature."""
    plot_df = df.copy()
    if cycle_min is not None and cycle_max is not None:
        plot_df = plot_df[(plot_df["cycle"] >= cycle_min) & (plot_df["cycle"] <= cycle_max)]
        
    fig = go.Figure()
    
    # Voltage trace (Primary axis)
    fig.add_trace(go.Scatter(
        x=plot_df["cycle"],
        y=plot_df["voltage_mean"],
        mode="lines",
        name="Mean Voltage (V)",
        line=dict(color=COLOR_ACCENT_TEAL, width=2.2),
        hovertemplate="Voltage: <b>%{y:.3f} V</b><extra></extra>"
    ))
    
    # Temperature trace (Secondary axis)
    fig.add_trace(go.Scatter(
        x=plot_df["cycle"],
        y=plot_df["temp_mean"],
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


def create_detailed_telemetry_multipanel(
    df: pd.DataFrame,
    cell_id: str,
    cycle_min: int = None,
    cycle_max: int = None
) -> go.Figure:
    """Creates a 3-panel synchronized time-series view for Voltage, Temperature, and Current."""
    plot_df = df.copy()
    if cycle_min is not None and cycle_max is not None:
        plot_df = plot_df[(plot_df["cycle"] >= cycle_min) & (plot_df["cycle"] <= cycle_max)]
        
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Terminal Voltage Dynamics (V)",
            "Thermal Dissipation Profile (°C)",
            "Discharge Current Profile (A)"
        ]
    )
    
    # Subplot 1: Voltage
    fig.add_trace(
        go.Scatter(x=plot_df["cycle"], y=plot_df["voltage_mean"], name="Mean Voltage (V)", line=dict(color=COLOR_ACCENT_TEAL, width=2)),
        row=1, col=1
    )
    if "voltage_max" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df["cycle"], y=plot_df["voltage_max"], name="Max Voltage (V)", line=dict(color=COLOR_ACCENT_SAGE, width=1, dash="dot")),
            row=1, col=1
        )
        
    # Subplot 2: Temperature
    fig.add_trace(
        go.Scatter(x=plot_df["cycle"], y=plot_df["temp_mean"], name="Mean Temp (°C)", line=dict(color=COLOR_WARNING, width=2)),
        row=2, col=1
    )
    if "temp_max" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df["cycle"], y=plot_df["temp_max"], name="Peak Temp (°C)", line=dict(color=COLOR_CRITICAL, width=1, dash="dot")),
            row=2, col=1
        )
        
    # Subplot 3: Current
    if "current_mean" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df["cycle"], y=plot_df["current_mean"], name="Mean Current (A)", line=dict(color=COLOR_INK, width=2)),
            row=3, col=1
        )

    fig.update_layout(
        height=620,
        margin=dict(l=55, r=40, t=45, b=45),
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        font=dict(family=FONT_FAMILY, color=COLOR_SLATE, size=11),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center")
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    fig.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER)
    fig.update_xaxes(title_text="Discharge Cycle", row=3, col=1)
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


def create_feature_trends_chart(df_feat: pd.DataFrame, cell_id: str) -> go.Figure:
    """Creates a chart of extracted physical degradation features over cycle aging."""
    fig = go.Figure()
    
    # 1. Rolling Decay Rate
    if "rolling_decay_rate" in df_feat.columns:
        fig.add_trace(go.Scatter(
            x=df_feat["cycle"],
            y=df_feat["rolling_decay_rate"],
            mode="lines",
            name="Rolling Decay Rate (ΔQ/5 cycles)",
            line=dict(color=COLOR_ACCENT_TEAL, width=2),
            hovertemplate="Decay Rate: <b>%{y:.5f} Ah/cyc</b><extra></extra>"
        ))
        
    # 2. Thermal Rise Above Ambient
    if "temp_rise" in df_feat.columns:
        fig.add_trace(go.Scatter(
            x=df_feat["cycle"],
            y=df_feat["temp_rise"],
            mode="lines",
            name="Thermal Rise Above Ambient (ΔT °C)",
            line=dict(color=COLOR_WARNING, width=2),
            yaxis="y2",
            hovertemplate="Temp Rise: <b>%{y:.2f} °C</b><extra></extra>"
        ))
        
    apply_industrial_layout(
        fig,
        title_text=f"Extracted Physical Degradation Dynamics — Cell {cell_id}",
        x_title="Cycle Number",
        y_title="Rolling Decay Rate (Ah/cycle)"
    )
    
    fig.update_layout(
        yaxis=dict(
            title=dict(text="Decay Rate (Ah/cycle)", font=dict(family=FONT_FAMILY, size=12, color=COLOR_ACCENT_TEAL)),
            tickfont=dict(family=MONO_FONT, color=COLOR_ACCENT_TEAL, size=11),
            gridcolor=COLOR_GRID
        ),
        yaxis2=dict(
            title=dict(text="Thermal Rise ΔT (°C)", font=dict(family=FONT_FAMILY, size=12, color=COLOR_WARNING)),
            tickfont=dict(family=MONO_FONT, color=COLOR_WARNING, size=11),
            overlaying="y",
            side="right",
            showgrid=False
        )
    )
    return fig


def create_feature_scatter_chart(df_feat: pd.DataFrame, cell_id: str) -> go.Figure:
    """Creates a feature relationship scatter of Capacity vs Voltage Range colored by Peak Temp."""
    fig = go.Figure()
    
    color_var = df_feat["temp_max"] if "temp_max" in df_feat.columns else df_feat["cycle"]
    x_var = df_feat["voltage_range"] if "voltage_range" in df_feat.columns else df_feat["voltage_mean"]
    x_label = "Discharge Voltage Range (Vmax - Vmin)" if "voltage_range" in df_feat.columns else "Mean Voltage (V)"
    
    fig.add_trace(go.Scatter(
        x=x_var,
        y=df_feat["capacity"],
        mode="markers",
        marker=dict(
            size=6,
            color=color_var,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=dict(text="Peak Temp (°C)", font=dict(family=FONT_FAMILY, size=11)))
        ),
        hovertemplate="Voltage Range: %{x:.3f} V<br>Capacity: %{y:.3f} Ah<extra></extra>"
    ))
    
    apply_industrial_layout(
        fig,
        title_text=f"Feature Correlation: Capacity vs {x_label} — Cell {cell_id}",
        x_title=x_label,
        y_title="Capacity (Ah)"
    )
    return fig


def create_model_comparison_bar_chart(comp_df: pd.DataFrame, cell_id: str) -> go.Figure:
    """Creates a bar chart comparing Test RMSE across PINN, AI, and Physics models."""
    fig = go.Figure()
    
    cell_data = comp_df[comp_df["Cell"] == cell_id]
    if cell_data.empty:
        cell_data = comp_df.groupby("Model", as_index=False)["RMSE"].mean()
        title_suffix = "Benchmark Fleet Average"
    else:
        title_suffix = f"Cell {cell_id}"
        
    colors = []
    for m in cell_data["Model"]:
        m_l = str(m).lower()
        if "pinn" in m_l:
            colors.append(COLOR_ACCENT_TEAL)
        elif "ai" in m_l:
            colors.append(COLOR_AI_BASELINE)
        else:
            colors.append(COLOR_PHYSICS_BASELINE)
            
    fig.add_trace(go.Bar(
        x=cell_data["Model"],
        y=cell_data["RMSE"],
        marker_color=colors,
        text=[f"{v:.4f}" for v in cell_data["RMSE"]],
        textposition="outside",
        textfont=dict(family=MONO_FONT, size=11, color=COLOR_INK),
        hovertemplate="Model: <b>%{x}</b><br>Test RMSE: <b>%{y:.4f} Ah</b><extra></extra>"
    ))
    
    apply_industrial_layout(
        fig,
        title_text=f"Controlled Test RMSE Benchmark Comparison — {title_suffix}",
        x_title="Prognostic Model",
        y_title="Test Root Mean Square Error (RMSE)"
    )
    return fig
