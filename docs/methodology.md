# Research Methodology

## 1. Scientific Objective

The overarching research objective is to empirically evaluate whether embedding governing battery degradation physics and monotonicity constraints into neural network training (PINN) yields measurable gains over purely empirical physics equations (Physics-Only) and purely statistical sequence models (AI-Only).

## 2. Experimental Hypotheses (H1-H4)

- **H1 (Predictive Accuracy):** Hybrid PINN achieves statistically significant reductions in RMSE, MAE, and MAPE and higher R2 across test cycles compared to Physics-Only and AI-Only baselines.
- **H2 (Physical Plausibility):** Hybrid PINN enforces continuous physical realism, dramatically reducing monotonicity violation rates and eliminating negative capacity artifacts.
- **H3 (Long-Horizon Generalization):** Autoregressive rollout predictions for horizons > 50 cycles remain stable and bounded under PINN, whereas pure AI models drift or diverge.
- **H4 (Low-Data Sample Efficiency):** When training history is restricted to 20%, 40%, 60%, and 80%, PINN retains higher prognostic fidelity than AI-Only due to the inductive bias provided by physics loss.

## 3. Data Integrity & Leakage Prevention Protocol

To ensure research validity:
1. **No Data Leakage Across Splits:** All feature scaling (StandardScaler) is strictly computed on training sets only and applied unchanged to validation/test sets.
2. **Strict Chronological Sequence Splitting:** For temporal splits, cycles are partitioned chronologically (Train: 0-60%, Val: 60-75%, Test: 75-100%).
3. **Identical Experimental Conditions:** All three models (Physics, AI, PINN) evaluate on the exact same cell instances, cycle ranges, and normalized targets.

## 4. Evaluation Metrics Matrix

1. **Accuracy Metrics:**
   - Root Mean Squared Error (RMSE): sqrt((1/N) * sum((y_i - y_hat_i)^2))
   - Mean Absolute Error (MAE): (1/N) * sum(|y_i - y_hat_i|)
   - Mean Absolute Percentage Error (MAPE): (1/N) * sum(|(y_i - y_hat_i) / y_i|) * 100%
   - Coefficient of Determination (R2): 1 - (sum((y_i - y_hat_i)^2) / sum((y_i - y_mean)^2))

2. **Physical Consistency Metrics:**
   - Monotonicity Violation Rate (% of steps where y_hat_{t+1} > y_hat_t)
   - Negative Capacity Rate (% of steps where y_hat_t <= 0)
   - Discontinuity / Gradient Spike Rate

3. **Uncertainty Calibration Metrics:**
   - Prediction Interval Coverage Probability (PICP) for nominal 90% and 95% intervals
   - Mean Prediction Interval Width (MPIW)
