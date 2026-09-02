# Research Limitations & Assumptions

## 1. Dataset & Operational Context
- **Fixed Ambient Conditions:** NASA PCoE experiments are conducted at fixed ambient room temperature (~24C) with CC-CV charging; real EV/BMS environments feature dynamic temperatures and regenerative braking.
- **Chemistry Specificity:** Datasets primarily focus on LiCoO2 / graphite cells; NMC, LFP, and silicon-anode blends may exhibit distinct phase transitions.

## 2. Physics Model Approximations
- Empirical and semi-empirical ODE equations approximate microscopic electrochemical kinetics (e.g. Doyle-Fuller-Newman / P2D models) without requiring internal concentration PDEs.
- Parameter fitting assumes baseline initial condition stability.

## 3. Computational & Uncertainty Assumptions
- Monte Carlo Dropout captures epistemic model uncertainty under Gaussian dropout assumptions.
- Long-horizon multi-step autoregression accumulates recursive single-step errors.
