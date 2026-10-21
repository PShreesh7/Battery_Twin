"""
Industrial Battery Digital Twin Physical Cell Visualization Component.
Renders an SVG/HTML technical schematic of an industrial 18650 cylindrical cell with live telemetry overlays.
"""

def render_battery_digital_twin_card(
    cell_id: str,
    soh_pct: float,
    current_cap: float,
    init_cap: float,
    voltage_mean: float,
    temp_mean: float,
    status_label: str
) -> str:
    # Color selection based on health
    if soh_pct >= 90.0:
        fill_color = "#23845B"
        glow_color = "rgba(35, 132, 91, 0.25)"
        status_bg = "#E8F5EE"
        status_text = "#1B6344"
        sei_status = "STABLE PASSIVATION LAYER"
    elif soh_pct >= 80.0:
        fill_color = "#B7791F"
        glow_color = "rgba(183, 121, 31, 0.25)"
        status_bg = "#FEF6E9"
        status_text = "#8C5A14"
        sei_status = "ACCELERATED SEI GROWTH"
    else:
        fill_color = "#C63D4F"
        glow_color = "rgba(198, 61, 79, 0.25)"
        status_bg = "#FDECEE"
        status_text = "#932938"
        sei_status = "CRITICAL SEI BREAKDOWN / PLATING RISK"

    # Clamp visual fill width between 5% and 100%
    visual_fill = max(6.0, min(100.0, soh_pct))
    bar_width_px = int(220 * (visual_fill / 100.0))

    html = f"""
    <div style="
        background: #FFFFFF;
        border: 1px solid #DCE4E9;
        border-radius: 8px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(23, 43, 58, 0.04);
        margin-bottom: 20px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #087F8C; letter-spacing: 0.08em; text-transform: uppercase;">
                    PHYSICAL DIGITAL TWIN TWIN-STATE
                </div>
                <div style="font-size: 16px; font-weight: 700; color: #172B3A; letter-spacing: -0.01em;">
                    LiCoO₂ Cylindrical 18650 Cell Simulation
                </div>
            </div>
            <div style="
                background: {status_bg};
                color: {status_text};
                border: 1px solid {fill_color}40;
                padding: 4px 12px;
                border-radius: 6px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.05em;
            ">
                ● {status_label}
            </div>
        </div>

        <div style="display: flex; gap: 32px; align-items: center; flex-wrap: wrap;">
            <!-- SVG Technical Battery Rendering -->
            <div style="flex: 0 0 320px; text-align: center; padding: 10px 0;">
                <svg width="300" height="120" viewBox="0 0 300 120" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="canisterGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stop-color="#EBF1F5" />
                            <stop offset="50%" stop-color="#FFFFFF" />
                            <stop offset="100%" stop-color="#DCE4E9" />
                        </linearGradient>
                        <linearGradient id="fillGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stop-color="{fill_color}99" />
                            <stop offset="100%" stop-color="{fill_color}" />
                        </linearGradient>
                        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                            <feGaussianBlur stdDeviation="3" result="blur" />
                            <feComposite in="SourceGraphic" in2="blur" operator="over" />
                        </filter>
                    </defs>

                    <!-- Outer Steel Canister Body -->
                    <rect x="25" y="20" width="240" height="80" rx="8" fill="url(#canisterGrad)" stroke="#172B3A" stroke-width="2.2" />
                    
                    <!-- Negative terminal base crimp -->
                    <rect x="18" y="32" width="7" height="56" rx="2" fill="#607180" stroke="#172B3A" stroke-width="1.5" />
                    
                    <!-- Positive terminal button (+) -->
                    <rect x="265" y="42" width="14" height="36" rx="3" fill="#DCE4E9" stroke="#172B3A" stroke-width="2" />
                    <text x="272" y="63" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" fill="#172B3A" text-anchor="middle">+</text>
                    <text x="12" y="64" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" fill="#607180" text-anchor="middle">-</text>

                    <!-- Internal Capacity Fill Bar -->
                    <rect x="35" y="30" width="220" height="60" rx="4" fill="#F3F6F8" stroke="#DCE4E9" stroke-width="1" />
                    <rect x="35" y="30" width="{bar_width_px}" height="60" rx="4" fill="url(#fillGrad)" />

                    <!-- Grid / SEI Layer Tick Marks inside fill -->
                    <line x1="80" y1="30" x2="80" y2="90" stroke="#FFFFFF" stroke-opacity="0.3" stroke-width="1.5" stroke-dasharray="2 2" />
                    <line x1="135" y1="30" x2="135" y2="90" stroke="#FFFFFF" stroke-opacity="0.3" stroke-width="1.5" stroke-dasharray="2 2" />
                    <line x1="190" y1="30" x2="190" y2="90" stroke="#FFFFFF" stroke-opacity="0.3" stroke-width="1.5" stroke-dasharray="2 2" />

                    <!-- 80% EOL Marker on cell -->
                    <line x1="211" y1="24" x2="211" y2="96" stroke="#C63D4F" stroke-width="2" stroke-dasharray="3 2" />
                    <text x="211" y="18" font-family="'JetBrains Mono', monospace" font-size="9" font-weight="700" fill="#C63D4F" text-anchor="middle">80% EOL</text>

                    <!-- Technical Label Overlay inside cell -->
                    <text x="145" y="65" font-family="'JetBrains Mono', monospace" font-size="15" font-weight="800" fill="#172B3A" text-anchor="middle">
                        {soh_pct:.1f}% SOH
                    </text>
                    
                    <!-- Dimension markings -->
                    <text x="145" y="112" font-family="'JetBrains Mono', monospace" font-size="9" fill="#607180" text-anchor="middle">
                        18650 CYLINDRICAL (18mm × 65mm) • CELL {cell_id}
                    </text>
                </svg>
            </div>

            <!-- Real-Time Telemetry Readout Grid -->
            <div style="flex: 1; min-width: 280px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px;">
                <div style="background: #F8FAFB; border: 1px solid #DCE4E9; border-radius: 6px; padding: 10px 14px;">
                    <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; color: #607180; letter-spacing: 0.05em;">Terminal Voltage</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; color: #172B3A; margin-top: 2px;">
                        {voltage_mean:.3f} <span style="font-size: 12px; font-weight: 500; color: #607180;">V</span>
                    </div>
                    <div style="font-size: 11px; color: #607180; margin-top: 2px;">Mean discharge potential</div>
                </div>

                <div style="background: #F8FAFB; border: 1px solid #DCE4E9; border-radius: 6px; padding: 10px 14px;">
                    <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; color: #607180; letter-spacing: 0.05em;">Core Temperature</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; color: #B7791F; margin-top: 2px;">
                        {temp_mean:.1f} <span style="font-size: 12px; font-weight: 500; color: #607180;">°C</span>
                    </div>
                    <div style="font-size: 11px; color: #607180; margin-top: 2px;">Thermal sensor telemetry</div>
                </div>

                <div style="background: #F8FAFB; border: 1px solid #DCE4E9; border-radius: 6px; padding: 10px 14px;">
                    <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; color: #607180; letter-spacing: 0.05em;">Remaining Capacity</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; color: #087F8C; margin-top: 2px;">
                        {current_cap:.3f} <span style="font-size: 12px; font-weight: 500; color: #607180;">Ah</span>
                    </div>
                    <div style="font-size: 11px; color: #607180; margin-top: 2px;">Baseline: {init_cap:.3f} Ah</div>
                </div>

                <div style="background: #F8FAFB; border: 1px solid #DCE4E9; border-radius: 6px; padding: 10px 14px;">
                    <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; color: #607180; letter-spacing: 0.05em;">SEI Interface Dynamic</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; color: #172B3A; margin-top: 4px;">
                        {sei_status}
                    </div>
                    <div style="font-size: 11px; color: #607180; margin-top: 2px;">Physics Arrhenius model</div>
                </div>
            </div>
        </div>
    </div>
    """
    return html
