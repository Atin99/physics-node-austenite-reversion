# Mathematical Supplement

Every equation, constant, and physical model used in this project, with derivations and literature sources.

**Code ↔ Equation Cross-Reference:**

| Section | Source File | Function |
|---|---|---|
| §1 Thermodynamics | `src/thermodynamics.py` | `get_Ac1_Ac3_fallback`, `get_equilibrium_RA_fallback`, `get_driving_force_fallback` |
| §2 Diffusion & Kinetics | `src/features.py` | `compute_diffusivity`, `compute_hollomon_jaffe`, `compute_JMAK`, `compute_k_arrhenius` |
| §3 Austenite Stability | `src/features.py` | `compute_Md30` |
| §4 Neural ODE | `src/model.py` | `AugmentedODEFunc`, `PhysicsNODE` |
| §5 Loss Functions | `src/losses.py` | `PhysicsConstrainedLoss` |
| §6 Uncertainty | `src/model.py` | `SWAG`, `predict_with_uncertainty` |

---

## 1. Thermodynamics

### 1.1 Critical Temperatures (Ac1, Ac3)

The Ac1 and Ac3 temperatures define the intercritical region where austenite and ferrite coexist. The standard Andrews (1965) correlations significantly overestimate Ac1 for medium-Mn steels because they were calibrated on low-alloy steels (Mn < 2 wt%). We recalibrated the coefficients against our 25-study literature dataset.

**Ac1 (recalibrated for medium-Mn):**

    Ac1 = 700 − 18.0·Mn − 16.9·Ni + 29.1·Si + 16.9·Cr + 8.0·Al       [°C]

    if Mn > 4 wt%:
        Ac1 = Ac1 − 4.5·(Mn − 4)^1.3

    Ac1 = max(Ac1, 350)

**Ac3:**

    Ac3 = 910 − 203·√C − 15.2·Ni + 44.7·Si − 35·Mn + 11·Cr + 35.0·Al   [°C]

    Ac3 = max(Ac3, Ac1 + 50)

**Key differences from Andrews (1965):**

| Parameter | Andrews | This work | Reason |
|---|---|---|---|
| Base Ac1 | 723 °C | 700 °C | Medium-Mn lowers Ac1 more than predicted |
| Mn coefficient | −10.7 | −18.0 | Mn stabilizes austenite more strongly above 4 wt% |
| Nonlinear Mn term | none | −4.5·(Mn−4)^1.3 | Captures accelerating Ac1 depression above 4 wt% Mn |

**Calibration evidence:**
- Nakada 2014 (6Mn): RA observed at 500°C → Ac1 < 500°C
- Gibbs 2011 (7.1Mn): RA observed at 575°C → Ac1 < 575°C
- Sun 2018 (12Mn): RA observed at 575°C → Ac1 < 575°C
- Hausman 2017 (6Mn): RA observed at 575°C → Ac1 < 575°C
- Arlazarov 2012 (5Mn): RA observed at 600°C → Ac1 < 600°C

**Source:** Andrews, K.W. "Empirical formulae for the calculation of some transformation temperatures." JISI 203 (1965): 721-727. Modified coefficients fitted to the 25-study medium-Mn dataset.

**Code:** `thermodynamics.py → get_Ac1_Ac3_fallback()`


### 1.2 Equilibrium Austenite Fraction (f_eq)

The equilibrium retained austenite fraction is not simply the lever rule fraction of austenite at the intercritical temperature. In medium-Mn steels, f_eq is limited by Mn partitioning during annealing — not all potential austenite nucleation sites achieve sufficient Mn enrichment for room-temperature stability. Typical literature values are 30-65%, not 100%.

**Maximum achievable RA:**

    f_max = 0.20 + 0.025·Mn + 0.80·C + 0.02·Al

    f_max = clip(f_max, 0.10, 0.75)

This gives: 5Mn → ~33%, 7Mn → ~38%, 9Mn → ~43%, 12Mn → ~50% (at C=0.10).

**Temperature dependence:**

Below Ac1:

    f_eq = 0

Intercritical region (Ac1 < T < Ac3):

    T_norm = (T − Ac1) / (Ac3 − Ac1)              [dimensionless, 0 to 1]

    f_eq = f_max · (3·T_norm² − 2·T_norm³)         [Hermite smoothstep]

The Hermite smoothstep S(x) = 3x² − 2x³ was chosen over a linear ramp because:
- S(0) = 0 and S(1) = 1 (boundary conditions satisfied)
- S'(0) = 0 and S'(1) = 0 (smooth onset and approach to saturation)
- This better matches the observed sigmoidal temperature dependence of RA

Above Ac3:

    T_over = (T − Ac3) / 100                       [dimensionless]
    f_eq = f_max · max(0.5, 1.0 − 0.15·T_over)

At temperatures above Ac3, coarsening reduces austenite stability, leading to decreased RA upon quenching.

**Mn enrichment in austenite:**

    K_Mn = 1.5 + 0.1·(Mn − 4)                     [partition coefficient]

    X_Mn^γ = Mn · K_Mn / (f_eq · K_Mn + (1 − f_eq))

This is the mass balance equation: the average Mn in austenite (X_Mn^γ) depends on the partition coefficient K_Mn = (X_Mn^γ / X_Mn^α) and the phase fraction.

**Code:** `thermodynamics.py → get_equilibrium_RA_fallback()`


### 1.3 Gibbs Free Energy Driving Force (ΔG)

The driving force for the α→γ transformation is approximated by a polynomial in temperature and composition:

    ΔG = −1462.4 + 8.282·T_K − 1.15×10⁻³·T_K² − 800·(Mn/100) − 22000·(C/100) + 3000·(Al/100)   [J/mol]

where T_K is temperature in Kelvin.

**Physical interpretation:**
- The T and T² terms represent the entropy contribution (austenite is stabilized at high T)
- The Mn and C terms are negative → these elements stabilize austenite (lower ΔG)
- The Al term is positive → Al stabilizes ferrite (raises ΔG)

**Note:** This is a simplified expression. A rigorous calculation would use CALPHAD (e.g., pycalphad with TCFE12 database) to compute ΔG^{BCC→FCC} at each composition and temperature. The code supports CALPHAD when available (`get_equilibrium_RA_calphad`).

**Code:** `thermodynamics.py → get_driving_force_fallback()`

---

## 2. Diffusion and Kinetics

### 2.1 Arrhenius Diffusivity

Manganese diffusion controls the kinetics of austenite reversion. The temperature dependence follows Arrhenius kinetics:

    D = D₀ · exp(−Q / (R·T))                       [m²/s]

| Species | D₀ [m²/s] | Q [J/mol] | Source |
|---|---|---|---|
| Mn in austenite (γ) | 1.785 × 10⁻⁵ | 264,000 | Porter & Easterling, "Phase Transformations in Metals and Alloys" |
| Mn in ferrite (α) | 7.56 × 10⁻⁵ | 224,500 | Same |
| C in austenite (γ) | 2.343 × 10⁻⁵ | 148,000 | Same |

**R** = 8.314 J/(mol·K) (universal gas constant)

**Example:** At T = 650°C = 923.15 K:

    D_Mn^γ = 1.785×10⁻⁵ · exp(−264000 / (8.314 × 923.15))
           = 1.785×10⁻⁵ · exp(−34.39)
           ≈ 2.06 × 10⁻²⁰ m²/s

This is consistent with measured Mn diffusivities in FCC iron at this temperature.

**Code:** `features.py → compute_diffusivity()`


### 2.2 Hollomon-Jaffe Parameter (P)

The Hollomon-Jaffe parameter is a classical tempering parameter that combines temperature and time into a single scalar representing the "thermal history" of the treatment:

    P = T_K · (C_HJ + log₁₀(t_hours))

where:
- T_K = temperature in Kelvin
- t_hours = time in hours
- C_HJ = 20 (Hollomon-Jaffe constant)

**Physical meaning:** P represents the total diffusion distance, combining the Arrhenius temperature dependence of diffusivity with the √t dependence of diffusion distance. Equal P values correspond to equivalent microstructural states regardless of the specific T-t combination.

**Normalization:** In the model input, P is normalized as P/20000.

**Source:** Hollomon, J.H. and Jaffe, L.D. "Time-temperature relations in tempering steel." Trans. AIME 162 (1945): 223-249.

**Code:** `features.py → compute_hollomon_jaffe()`


### 2.3 Johnson-Mehl-Avrami-Kolmogorov (JMAK) Kinetics

The JMAK equation describes the fraction transformed as a function of time under isothermal conditions, assuming nucleation-and-growth kinetics:

    f(t) = f_eq · (1 − exp(−k · t^n))

where:
- f_eq = equilibrium fraction (from §1.2)
- k = rate constant [s⁻ⁿ]
- n = Avrami exponent (related to nucleation and growth geometry)

**Avrami exponent ranges:**
- n = 1: 1D growth, no nucleation (diffusion-controlled thickening of existing films)
- n = 1.5: 1D growth, constant nucleation rate
- n = 2: 2D growth (plate-like), no nucleation
- n = 3: 3D growth (equiaxed), no nucleation
- n = 4: 3D growth, constant nucleation rate

For medium-Mn steels, n typically ranges from 1.5 to 3.0, consistent with mixed film/lath growth of austenite.

**JMAK rate (derivative):**

    df/dt = f_eq · k · n · t^(n−1) · exp(−k · t^n)

**Temperature-dependent rate constant (Arrhenius):**

    k(T) = k₀ · (X_Mn/7.0)^0.8 · (X_C/0.1)^0.3 · exp(−Q_eff / (R·T))

| Parameter | Value | Unit |
|---|---|---|
| k₀ | 1 × 10⁻⁸ | s⁻ⁿ |
| Q_eff | 200,000 | J/mol |
| n range | 1.5 - 3.0 | dimensionless |

The composition scaling (Mn^0.8, C^0.3) reflects the observation that higher Mn and C accelerate reversion kinetics through increased driving force and nucleation site density.

**Role in the model:** The JMAK equation is used for:
1. Generating synthetic training data (Stage 1)
2. Physics loss: penalizing early-time deviations from JMAK-like behavior

**Code:** `features.py → compute_JMAK()`, `compute_JMAK_rate()`, `compute_k_arrhenius()`

---

## 3. Austenite Mechanical Stability (Md30)

### 3.1 Angel's Md30 Formula

Md30 is the temperature at which 50% of the austenite transforms to martensite under 30% true strain. It quantifies the mechanical stability of retained austenite and hence its TRIP effectiveness.

    Md30 = 413 − 462·(C+N) − 9.2·Si − 8.1·Mn − 13.7·Cr − 9.5·Ni − 18.5·Mo   [°C]

**Interpretation:**
- Md30 > 80°C: austenite transforms easily → strong TRIP effect but potentially unstable
- Md30 = 20-60°C: optimal range for automotive applications (transforms during forming)
- Md30 < 0°C: mechanically stable austenite → weaker TRIP contribution

**Source:** Angel, T. "Formation of martensite in austenitic stainless steels." JISI 177 (1954): 165-174.

**Code:** `features.py → compute_Md30()`

---

## 4. Neural ODE Architecture

### 4.1 Latent ODE Formulation

The model learns the kinetics as a continuous-time dynamical system in a latent space:

    dz/dt = f_θ(z, c)                               [latent ODE]

where:
- z ∈ ℝ^(1+d_aug) is the latent state (first component = austenite fraction, remaining d_aug = 4 augmented dimensions)
- c is a conditioning vector encoding composition and thermodynamic features
- f_θ is a neural network parameterized by θ

The augmented dimensions (d_aug = 4) provide additional degrees of freedom for the ODE to represent complex dynamics that cannot be captured by a 1D ODE. They have no direct physical interpretation.

### 4.2 Composition Encoder

The alloy composition vector [T_norm, Mn, C, Al, Si] is encoded using a multi-head self-attention mechanism:

    x_i = W_embed · comp_i + E_element(i)            [per-element embedding]

    attn_out = MultiHeadAttention(X, X, X)            [4 heads, d=32]

    c_comp = LayerNorm(mean(attn_out))                [pooled composition embedding]

where:
- W_embed ∈ ℝ^(32×1) maps each scalar composition value to a 32-dim vector
- E_element is a learned embedding table for each element position
- The attention allows the model to learn element-element interactions (e.g., Mn-C synergy)

### 4.3 FiLM Conditioning

The thermodynamic features (log D, ΔG/1000, P/20000) are injected into the ODE function via Feature-wise Linear Modulation (FiLM):

    h_out = h · (1 + γ(c)) + β(c)

where γ and β are learned linear projections of the conditioning vector c. This is applied after each hidden layer in the ODE network, allowing the composition/temperature to modulate the kinetics at every level.

**Source:** Perez, E. et al. "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI (2018).

### 4.4 Physics-Gated ODE Vector Field

The ODE right-hand side for the austenite fraction is not a free neural network output. It is multiplicatively gated by three physics terms:

    df/dt = rate(z, c) · |ΔG| · saturation · nucleation

where:
- **rate(z, c)** = softplus(W·h + b) × 0.01 — learned, always non-negative
- **|ΔG|** = absolute driving force (+ 0.1 offset for numerical stability) — ensures no transformation without thermodynamic driving force
- **saturation** = clamp(f_eq − f, min=0) + ε — ensures transformation slows as equilibrium is approached
- **nucleation** = f + ε_nuc — captures the initial acceleration (autocatalytic nucleation, more existing austenite → more nucleation sites)

**ε_nuc** = 0.005 (nucleation regularization constant)

This decomposition guarantees:
- df/dt ≥ 0 (monotonicity — austenite cannot spontaneously decompose during isothermal hold)
- df/dt → 0 as f → f_eq (equilibrium boundary)
- df/dt = 0 when ΔG = 0 (no driving force = no transformation)

**Code:** `model.py → AugmentedODEFunc.forward()`

### 4.5 ODE Solver

The ODE is solved using the Dormand-Prince 4/5 adaptive-step Runge-Kutta method (`dopri5`):

| Parameter | Value | Meaning |
|---|---|---|
| Method | dopri5 | Dormand-Prince 4th/5th order |
| rtol | 1 × 10⁻⁵ | Relative tolerance |
| atol | 1 × 10⁻⁷ | Absolute tolerance |
| max_num_steps | 10,000 | Maximum steps per integration |

Training uses the adjoint method (`odeint_adjoint`) for memory-efficient backpropagation through the ODE. The adjoint method solves a backward ODE for the gradients, avoiding storing the full forward trajectory.

**Source:** Chen, R.T.Q. et al. "Neural Ordinary Differential Equations." NeurIPS (2018).

### 4.6 Output Clamping

After solving the ODE, the raw output is clamped to enforce physical bounds:

    f_out = min(max(f_raw, 0), f_eq × 1.02)

The 2% tolerance above f_eq allows slight overshoots during training to avoid gradient issues at the boundary.

---

## 5. Loss Function

### 5.1 Total Loss

    L = L_data + 0.1·L_physics + 0.5·L_mono + 0.5·L_bound

### 5.2 Data Loss (Huber + MSE blend)

    L_data = 0.7·L_Huber + 0.3·L_MSE

**Huber loss** (δ = 0.05):

    L_Huber = { 0.5·e²          if |e| ≤ δ
              { δ·(|e| − δ/2)   if |e| > δ

where e = f_pred − f_true.

**Rationale:** The Huber component provides robustness to outliers (digitized data points with ±5% error), while the MSE component ensures the loss surface is smooth near zero for fine-tuning.

### 5.3 Physics Loss (JMAK consistency)

    f_JMAK(t) = f_eq · (1 − exp(−k·t^n))

    L_physics = mean[(f_pred(t) − f_JMAK(t))² · M_early]

where M_early is a mask selecting only the early portion of the curve (t ≤ 0.5·t_max). This penalizes early-time deviations from expected JMAK behavior while allowing the Neural ODE to deviate at later times where multi-mechanism effects may operate.

### 5.4 Monotonicity Loss

    L_mono = mean[ReLU(−Δf)²]

where Δf = f(t_{i+1}) − f(t_i) for consecutive time steps. This penalizes any decrease in austenite fraction, enforcing the physical constraint that austenite reversion is irreversible under isothermal conditions.

### 5.5 Boundary Loss

    L_bound = mean[ReLU(f_pred − f_eq)²]

Penalizes predictions that exceed the equilibrium fraction. The ReLU ensures no penalty when f_pred ≤ f_eq.

### 5.6 Adaptive Loss Weighting (optional)

Two optional mechanisms for automatic loss balancing:

**Homoscedastic uncertainty** (Kendall et al., 2018):

    L_total = Σᵢ exp(−sᵢ)·Lᵢ + sᵢ

where sᵢ = log σᵢ² are learned log-variance parameters for each loss term.

**GradNorm** (Chen et al., 2018):

    wᵢ(t+1) = wᵢ(t) · exp(−lr · sign(Gᵢ − G̅ · rᵢ^α))

where Gᵢ is the gradient norm of loss i with respect to the shared layer, rᵢ is the relative training rate, and α = 1.5 controls the strength of balancing.

**Code:** `losses.py → PhysicsConstrainedLoss`

---

## 6. Uncertainty Quantification

### 6.1 MC Dropout

During inference, dropout is kept active and the prediction is repeated n_samples = 100 times. The mean and quantiles of the resulting distribution provide a measure of epistemic uncertainty:

    f̄ = (1/N) Σᵢ f_θ^(i)                            [mean prediction]

    f_lower = quantile(f_θ^(i), α/2)                  [lower CI bound]
    f_upper = quantile(f_θ^(i), 1−α/2)                [upper CI bound]

where α = 0.05 for a 95% confidence interval.

### 6.2 SWAG (Stochastic Weight Averaging - Gaussian)

SWAG constructs an approximate posterior over the model weights by collecting weight snapshots during training:

**First moment (mean):**

    θ̄ = (1/T) Σₜ θₜ

**Second moment (diagonal variance):**

    σ² = (1/T) Σₜ θₜ² − θ̄²

**Low-rank deviation matrix:**

    D = [θ_{T-K+1} − θ̄, ..., θ_T − θ̄]              [K = 20 columns]

**Sampling:**

    θ_new = θ̄ + σ·z₁ + (1/√(2(K−1)))·D·z₂

where z₁ ~ N(0, I) and z₂ ~ N(0, I_K).

**Source:** Maddox, W.J. et al. "A Simple Baseline for Bayesian Deep Learning." NeurIPS (2019).

**Code:** `model.py → SWAG`

---

## 7. Input Feature Normalization

The raw inputs are normalized before entering the model:

| Feature | Raw | Normalized | Code |
|---|---|---|---|
| Temperature | T [K] | (T − 673) / 100 | T_ref = 673 K |
| Mn | wt% | wt% (unnormalized) | — |
| C | wt% | wt% (unnormalized) | — |
| Al | wt% | wt% (unnormalized) | — |
| Si | wt% | wt% (unnormalized) | — |
| Diffusivity | D [m²/s] | log₁₀(D + 10⁻³⁰) | Log-scale for ~20 orders of magnitude range |
| Driving force | ΔG [J/mol] | ΔG / 1000 | Scale to ~O(1) |
| Hollomon-Jaffe | P | P / 20000 | Scale to ~O(1) |

**Code:** `features.py → featurize_sample()`

---

## 8. Summary of All Physical Constants

| Constant | Symbol | Value | Unit | Source |
|---|---|---|---|---|
| Gas constant | R | 8.314 | J/(mol·K) | CODATA |
| Mn pre-exponential (γ) | D₀ | 1.785 × 10⁻⁵ | m²/s | Porter & Easterling |
| Mn activation energy (γ) | Q | 264,000 | J/mol | Porter & Easterling |
| Mn pre-exponential (α) | D₀ | 7.56 × 10⁻⁵ | m²/s | Porter & Easterling |
| Mn activation energy (α) | Q | 224,500 | J/mol | Porter & Easterling |
| C pre-exponential (γ) | D₀ | 2.343 × 10⁻⁵ | m²/s | Porter & Easterling |
| C activation energy (γ) | Q | 148,000 | J/mol | Porter & Easterling |
| Hollomon-Jaffe constant | C_HJ | 20 | — | Hollomon & Jaffe 1945 |
| Nucleation regularizer | ε_nuc | 0.005 | — | This work |
| JMAK pre-exponential | k₀ | 1 × 10⁻⁸ | s⁻ⁿ | Fitted to literature |
| JMAK effective Q | Q_eff | 200,000 | J/mol | Fitted to literature |
| Angel Md30 intercept | — | 413 | °C | Angel 1954 |

---

## 9. Notation Reference

| Symbol | Meaning |
|---|---|
| f, f_RA | Retained austenite volume fraction (0–1) |
| f_eq | Equilibrium austenite fraction |
| T, T_K, T_c | Temperature (K or °C as subscripted) |
| Ac1, Ac3 | Lower and upper intercritical temperatures |
| ΔG | Gibbs free energy driving force for α→γ |
| D | Diffusivity |
| P | Hollomon-Jaffe parameter |
| k, n | JMAK rate constant and Avrami exponent |
| Md30 | Temperature for 50% strain-induced transformation at 30% strain |
| z | Latent state vector of the Neural ODE |
| θ | Neural network parameters |
| c | Conditioning vector |
| L | Loss function |
