# Experiment Plan

## 1. Dataset Matrix

| Dataset | Battery Cell ID | Chemistry | Nominal Capacity | Cycles | Primary Use Case |
|---|---|---|---|---|---|
| NASA PCoE | B0005 | LiCoO2 / Graphite | 2.0 Ah | 168 | Primary Train / Benchmark |
| NASA PCoE | B0006 | LiCoO2 / Graphite | 2.0 Ah | 168 | Cross-Cell Evaluation |
| NASA PCoE | B0007 | LiCoO2 / Graphite | 2.0 Ah | 168 | Cross-Cell Evaluation |
| NASA PCoE | B0018 | LiCoO2 / Graphite | 2.0 Ah | 132 | Unseen-Cell Test Holdout |
| CALCE CS2 | CS2_35 | LiCoO2 / Graphite | 1.1 Ah | 800+ | Extended Cycle Benchmark |
| CALCE CS2 | CS2_36 | LiCoO2 / Graphite | 1.1 Ah | 800+ | Cross-Cell Evaluation |
| CALCE CS2 | CS2_37 | LiCoO2 / Graphite | 1.1 Ah | 800+ | Cross-Cell Evaluation |
| CALCE CS2 | CS2_38 | LiCoO2 / Graphite | 1.1 Ah | 800+ | Unseen-Cell Test Holdout |

## 2. Planned Experiments

### Experiment 1: Three-Model Baseline Benchmark (H1 & H2)
- Models: Physics-Only vs AI-Only LSTM vs Hybrid PINN.
- Splits: NASA B0005 (0-60% Train, 60-75% Val, 75-100% Test) and CALCE CS2_35.
- Target Outputs: 
esults/metrics/model_comparison.csv, figure comparison plots.

### Experiment 2: Long-Horizon Forecasting Stability (H3)
- Horizons: Short (10 cycles), Medium (30 cycles), Long (50+ cycles).
- Metrics: RMSE drift vs horizon, trajectory curvature.

### Experiment 3: Low-Data Efficiency Benchmark (H4)
- Subsets: 20%, 40%, 60%, 80% of training cycle history.
- Target Outputs: 
esults/metrics/data_efficiency_comparison.csv, sample efficiency curves.

### Experiment 4: Physics Weight Sensitivity Ablation
- Weight lambda_physics in [0, 0.001, 0.01, 0.1, 1.0, 10.0].
- Target Outputs: 
esults/metrics/table_ablation.csv.

### Experiment 5: Uncertainty Quantification & Calibration
- Method: Monte Carlo Dropout with T=100 stochastic forward passes.
- Verification: Empirical 90% and 95% PICP calibration curves.
