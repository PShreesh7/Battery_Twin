# ?? Battery Twin: Physics-Informed Neural Network Digital Twin

> **Physics-Informed Neural Network based Lithium-Ion Battery Health and Degradation Prediction Digital Twin**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ?? 1. Overview & Research Scope

Lithium-Ion batteries undergo capacity fading and impedance rise under repeated charge-discharge cycles due to Solid Electrolyte Interphase (SEI) growth and active material degradation. 

**Battery Twin** is a research-grade Digital Twin framework that implements and benchmarks three paradigms:
1. **Physics-Only Baseline:** Empirical degradation physics equations (Power Law, Exponential, Modified Logistic).
2. **AI-Only Baseline:** Purely data-driven sequence model (LSTM) trained on empirical measurement signals without physical inductive bias.
3. **Hybrid PINN (Physics-Informed Neural Network):** Dual-loss neural network embedding empirical degradation kinetics and monotonicity constraints directly into the loss function:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{physics}} \left( \mathcal{L}_{\text{residual}} + \gamma \mathcal{L}_{\text{monotonicity}} \right)$$

---

## ?? 2. Benchmark Datasets

| Dataset | Battery Cell IDs | Chemistry | Nominal Capacity | Primary Operating Conditions |
|---|---|---|---|---|
| **NASA Ames PCoE** | B0005, B0006, B0007, B0018 | LiCoO2 / Graphite | 2.0 Ah | CC-CV charge (1.5A to 4.2V), 2.0A CC discharge to 2.7V at ~24?C |
| **CALCE CS2** | CS2_35, CS2_36, CS2_37, CS2_38 | LiCoO2 / Graphite | 1.1 Ah | Standard CC-CV charge (0.5C), 0.5C discharge to 2.7V |

---

## ??? 3. Architecture & Project Layout

```text
Battery_Twin/
??? configs/               # Global & dataset YAML configurations
??? data/
?   ??? raw/               # Immutable raw battery datasets (NASA & CALCE)
?   ??? interim/           # Aligned time-series structures
?   ??? processed/         # Standardized cycle-level features
??? src/
?   ??? data/              # Ingestion loaders & quality validators
?   ??? preprocessing/     # Scalers (zero-leakage fit on train)
?   ??? features/          # Voltage, current, temperature feature extractors
?   ??? physics/           # Governing equations, curve fitting & loss modules
?   ??? models/            # AI-LSTM and PINN-LSTM architectures
?   ??? training/          # Unified trainer with early stopping & trackers
?   ??? evaluation/        # Metrics & Physical Plausibility Engine
?   ??? uncertainty/       # Monte Carlo Dropout & calibration curves
?   ??? inference/         # Production-ready stateless inference pipeline
??? scripts/               # Standalone execution & training scripts
??? artifacts/             # Serialized model weights, scalers, and configs
??? results/
?   ??? metrics/           # CSV benchmark tables
?   ??? figures/           # High-resolution publication plots
?   ??? reports/           # Comprehensive research reports
??? api/                   # FastAPI REST service & Pydantic schemas
??? dashboard/             # Interactive Streamlit + Plotly Digital Twin
??? tests/                 # Full Pytest unit and integration test suite
```

---

## ? 4. Quick Start & Installation

### Step 1: Clone and Set Up Virtual Environment
```bash
git clone https://github.com/PShreesh7/Battery_Twin.git
cd Battery_Twin

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scriptsctivate
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ?? 5. One-Command Full Pipeline Execution

Run the complete pipeline from scratch (Data Ingestion $\to$ Preprocessing $\to$ Physics/AI/PINN Training $\to$ Benchmark Evaluation $\to$ Forecasting $\to$ Low-Data $\to$ Ablation $\to$ Uncertainty):

```bash
python scripts/run_all_experiments.py
```

### Running Individual Stages:
- **Acquire Datasets:** `python scripts/download_data.py`
- **Validate Data Quality:** `python scripts/validate_data.py`
- **Preprocess & EDA:** `python scripts/preprocess.py`
- **Train Physics Baseline:** `python scripts/train_physics.py`
- **Train AI-Only Baseline:** `python scripts/train_ai.py`
- **Train Hybrid PINN:** `python scripts/train_pinn.py`
- **Evaluate Benchmarks:** `python scripts/evaluate.py`
- **Long-Horizon Forecasting:** `python scripts/forecast_experiment.py`
- **Low-Data Efficiency Benchmark:** `python scripts/low_data_experiment.py`
- **Physics Loss Ablation:** `python scripts/ablation_study.py`
- **Uncertainty Calibration:** `python scripts/uncertainty_experiment.py`

---

## ?? 6. Key Benchmark Findings

### 6.1 Long-Horizon Forecasting Stability (NASA B0005)
| Horizon | AI-Only (LSTM) RMSE | Hybrid PINN RMSE | PINN Error Reduction |
|---|---|---|---|
| **Short (10 cycles)** | 0.1063 Ah | **0.0498 Ah** | **53.1%** |
| **Medium (30 cycles)** | 0.0872 Ah | **0.0405 Ah** | **53.5%** |
| **Long (50 cycles)** | 0.0700 Ah | **0.0340 Ah** ($R^2=0.21$) | **51.4%** |

### 6.2 Low-Data Sample Efficiency
- When restricted to only **20% of training history**, Hybrid PINN achieves **0.1133 Ah RMSE** vs. AI-Only **0.1615 Ah RMSE** (**29.8% error reduction**).

### 6.3 Physical Plausibility
- Increasing $\lambda_{\text{physics}}$ systematically eliminates non-physical capacity jumps and reduces monotonicity violations from 5 down to 1.

---

## ?? 7. Deploying the REST API

Launch the FastAPI inference engine:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### Key Endpoints:
- `GET /health` - Service health status
- `GET /models` - List available models & supported cells
- `GET /battery/{battery_id}` - Metadata & EOL capacity threshold
- `POST /predict/capacity` - Predict single-cycle capacity
- `POST /predict/soh` - Estimate current State of Health (SOH)
- `POST /forecast` - Autoregressive multi-step degradation forecast
- `POST /uncertainty` - Monte Carlo Dropout confidence intervals
- `POST /physics-check` - Quantitative physical plausibility validation

---

## ?? 8. Interactive Digital Twin Dashboard

Launch the Streamlit Digital Twin application:
```bash
streamlit run dashboard/app.py
```
Open `http://localhost:8501` in your browser to interact with:
1. **Overview & Health:** Real-time SOH badge, EOL indicators, and physical invariant checks.
2. **Historical Telemetry:** Multi-sensor voltage, current, and temperature evolutions.
3. **Forecast & Uncertainty:** Shaded 90%/95% confidence bands and EOL horizon estimation.
4. **Three-Model Comparison:** Side-by-side metric tables and overlay plots.
5. **What-If Simulator:** Interactive scenario testing under customizable forecast horizons.

---

## ?? 9. Running Automated Tests

Run the complete test suite:
```bash
pytest -v
```

---

## ?? 10. License & Citations
This project is open-source under the [MIT License](LICENSE).
