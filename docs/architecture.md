# Battery Twin System Architecture

## 1. High-Level Overview

The **Battery Twin** is a modular, research-grade Physics-Informed Neural Network (PINN) digital twin for Lithium-Ion battery health prognostics and State of Health (SOH) estimation.

`	ext
+-----------------------------------------------------------------------------+
|                             DATA INGESTION LAYER                            |
|  +-------------------------------+     +---------------------------------+  |
|  | NASA PCoE Dataset (B0005...)  |     | CALCE CS2 Dataset (CS2_35...)   |  |
|  +---------------+---------------+     +----------------+----------------+  |
|                  +-----------------------+--------------+                   |
|                                          |                                  |
|                       +------------------v------------------+               |
|                       | Data Validation & Quality Checks    |               |
|                       | (Missing, NaNs, Outliers, Jumps)    |               |
|                       +------------------+------------------+               |
|                                          |                                  |
|                       +------------------v------------------+               |
|                       | Preprocessing & Feature Engineering |               |
|                       | (Voltage, Current, Temp Statistics) |               |
|                       +------------------+------------------+               |
|                                          |                                  |
|                       +------------------v------------------+               |
|                       | Unified Dataset & Temporal Splitting|               |
|                       | (Zero Leakage, Scaler Fitted on Train)              |
|                       +------------------+------------------+               |
+------------------------------------------+----------------------------------+
                                           |
+------------------------------------------+----------------------------------+
|                             MODELING & BENCHMARK                            |
|           +------------------------------+----------------------------+     |
|           |                              |                            |     |
|  +--------v--------+           +---------v--------+         +---------v-------+
|  |  Physics-Only   |           |     AI-Only      |         |   Hybrid PINN   |
|  | Empirical Fits  |           |   LSTM Network   |         | Dual-Loss LSTM  |
|  | (PowerLaw/Exp)  |           |  Pure Data Loss  |         | Data + Physics  |
|  +--------+--------+           +---------+--------+         +---------+-------+
|           +------------------------------+----------------------------+     |
|                                          |                                  |
|                       +------------------v------------------+               |
|                       | Fair Benchmark & Plausibility Engine|               |
|                       | (RMSE, MAE, R2, Monotonicity, Jump) |               |
|                       +------------------+------------------+               |
+------------------------------------------+----------------------------------+
                                           |
+------------------------------------------+----------------------------------+
|                           DEPLOYMENT & INFERENCE                            |
|                                          |                                  |
|                       +------------------v------------------+               |
|                       | Packaged Model Artifacts & Scalers  |               |
|                       +------------------+------------------+               |
|                                          |                                  |
|                       +------------------v------------------+               |
|                       | FastAPI REST Inference Engine       |               |
|                       | (SOH, Capacity, Forecast, UQ)       |               |
|                       +------------------+------------------+               |
|                                          |                                  |
|                       +------------------v------------------+               |
|                       | Streamlit Digital Twin Dashboard    |               |
|                       | (Plotly Interactive Visualizations) |               |
|                       +-------------------------------------+               |
+-----------------------------------------------------------------------------+
`

## 2. Component Specifications

### 2.1 Data Layer (src/data/, src/preprocessing/, src/features/)
- **Loaders:** NASADataLoader parses MATLAB .mat structures; CALCEDataLoader parses Excel/CSV multi-channel records.
- **Validator:** Verifies structural completeness, timestamps, monotonic cycle indices, signal boundary conditions, and generates dataset_quality_report.json.
- **Feature Extractor:** Extracts windowed sequence statistics (mean, max, min, standard deviation, skewness) across voltage, current, and temperature, alongside cycle index.
- **Unified Splitter:** Enforces strict temporal splitting (Train: 0-60%, Val: 60-75%, Test: 75-100%) and unseen-cell holdouts without data leakage.

### 2.2 Physics Engine (src/physics/)
- Implements closed-form empirical degradation dynamics:
  1. Power Law: Q(t) = Q_0 - alpha * t^beta
  2. Exponential: Q(t) = Q_0 * exp(-alpha * t) + beta
  3. Modified Logistic: Q(t) = Q_0 / (1 + alpha * exp(beta * t))
- Fits optimal parameters theta* on training data using non-linear least squares (scipy.optimize.curve_fit).
- Computes PyTorch-differentiable physics residuals:
  Residual(Q_hat, t; theta*) = Q_hat - Q_phys(t; theta*)
  along with soft-penalty monotonicity constraints max(0, Q_hat_{t+1} - Q_hat_t).

### 2.3 Model Layer (src/models/, src/training/)
- **AI-Only LSTM:** Multi-layer LSTM encoder with dropout and linear projection head, trained purely with MSE loss:
  L_data = (1/N) * sum((Q_hat_i - Q_i)^2)
- **Hybrid PINN LSTM:** Identical architecture to ensure fair comparison, optimized with the composite objective:
  L_total = L_data + lambda_physics * [ L_residual + gamma * L_monotonicity ]
- **Training Harness:** Automatic seed management, gradient clipping, ReduceLROnPlateau scheduling, early stopping on validation loss, and full metric history persistence.

### 2.4 Uncertainty & Evaluation Layer (src/uncertainty/, src/evaluation/)
- **Monte Carlo Dropout:** Runs T stochastic passes at test time with active dropout masks to estimate predictive mean and epistemic variance.
- **Calibration Engine:** Computes Prediction Interval Coverage Probability (PICP), Mean Prediction Interval Width (MPIW), and reliability diagrams.
- **Physical Plausibility Engine:** Computes quantitative violation rates for monotonicity, negative capacity, and unphysical gradient spikes, outputting structured validation statuses (PASS, WARNING, FAIL).

### 2.5 Inference & API Layer (src/inference/, pi/)
- High-performance, stateless inference pipeline deserializing models, scalers, and configs from rtifacts/.
- FastAPI endpoints for single-point SOH prediction, multi-step autoregressive forecasting, uncertainty intervals, and physics checks.

### 2.6 Digital Twin Dashboard (dashboard/)
- Interactive Streamlit application with live health diagnostics, historical cycle telemetry, interactive multi-horizon forecasts, three-model overlay comparisons, and what-if simulation sliders.
