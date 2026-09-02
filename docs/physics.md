# Degradation Physics & Mathematical Formulations

## 1. Lithium-Ion Battery Aging Mechanisms

Lithium-Ion battery capacity loss during continuous cycling is primarily governed by:
1. **Solid Electrolyte Interphase (SEI) Growth:** Continuous reaction between lithium ions and electrolyte solvent on the graphite negative electrode, causing Loss of Lithium Inventory (LLI).
2. **Active Material Loss (LAM):** Particle cracking and structural degradation in the cathode (e.g., LiCoO2) due to repeated volumetric expansion and contraction.
3. **Electrolyte Decomposition:** Impedance rise and internal resistance increase leading to usable capacity reduction under standard cutoff voltages.

## 2. Mathematical Equations

### 2.1 Power Law Formulation (Diffusion-Limited SEI Growth)
According to classical parabolic SEI growth kinetics:
Q(t) = Q_0 - \alpha \cdot t^\beta
where $ is initial capacity, $\alpha > 0$ is the degradation rate constant, and $\beta \approx 0.5$ (or .5 \le \beta \le 1.0$) reflects diffusion-limited growth.

### 2.2 Exponential Degradation Formulation
Q(t) = Q_0 \cdot \exp(-\alpha \cdot t) + \beta
where $\alpha$ represents the degradation decay constant.

### 2.3 Modified Verhulst / Logistic Model (Knee Point Aging)
Q(t) = \frac{Q_0}{1 + \alpha \cdot \exp(\beta \cdot t)}
capturing the steady initial phase followed by accelerated capacity rollover (aging knee).

## 3. Loss Formulation in PyTorch

The PINN loss is formulated as:
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{physics}} \mathcal{L}_{\text{physics}}
where:
\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{i=1}^N (\hat{Q}_i - Q_i)^2
\mathcal{L}_{\text{physics}} = \frac{1}{N} \sum_{i=1}^N (\hat{Q}_i - Q_{\text{phys}}(t_i; \boldsymbol{\theta}^*))^2 + \gamma \frac{1}{N-1} \sum_{i=1}^{N-1} \max(0, \hat{Q}_{i+1} - \hat{Q}_i)^2
This ensures gradients from physical laws directly propagate through the neural network parameters during backpropagation.
