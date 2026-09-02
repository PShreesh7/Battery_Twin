# Risk Register & Mitigation Strategy

| Risk ID | Risk Description | Severity | Likelihood | Mitigation Strategy |
|---|---|---|---|---|
| R1 | Inconsistent MATLAB / Excel raw data structures across dataset variants | High | Medium | Implement defensive schema parsing with automated type casting and cycle alignment in src/data/validator.py. |
| R2 | Numerical instability during curve fitting of non-linear physics parameters | Medium | Low | Use bounded Levenberg-Marquardt / Trust Region Reflective optimization in scipy.optimize.curve_fit with robust parameter bounds. |
| R3 | Data leakage during sequence normalization | High | Low | Fit StandardScaler strictly on training slices; serialize fitted scalers directly to rtifacts/. |
| R4 | Vanishing gradients or loss scale disparity between L_data and L_physics | Medium | Medium | Implement dynamic loss weighting and gradient clipping (max_norm=1.0) in trainer. |
| R5 | Autoregressive error compounding during 50+ cycle rollouts | Medium | High | Benchmark multi-step horizon metrics explicitly; use PINN physics regularization to constrain trajectory envelope. |
