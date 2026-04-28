# Physics-Constrained Neural ODE for Austenite Reversion Kinetics in Medium-Mn Steels

## Project Summary

This project tests how far a physics-informed kinetics model can go when trained on heterogeneous, partially incompatible literature data from medium-Mn steel studies.

The core question is not "can we build a predictor?" — it is:

> **When metallurgy knowledge exists across many papers but the data is fragmented and inconsistent, what actually limits predictive quality: the model architecture, the data quality, the target definition, or the mismatch between synthetic support and real experiments?**

### One-Line Summary

> I built a physics-informed Neural ODE pipeline for medium-Mn steel transformation kinetics, then used the actual training behavior to diagnose what limits predictive usefulness when working with messy literature data.

---

## What This Project Is

- A **physics-informed Neural ODE** for modeling austenite reversion / transformation kinetics
- Trained on **125 experimental data points from 25 peer-reviewed studies** (real literature data)
- Augmented with **calibrated and exploratory synthetic curves** to stabilize training
- Evaluated honestly on held-out real experimental data
- A diagnostic study of where the performance ceiling comes from

## What This Project Is NOT

- Not a production-ready predictor
- Not a claim that the model is "state of the art"
- Not AI magic applied to metallurgy
- Not a startup product demo

---

## Data Pipeline

### Three-Tier Architecture

| Tier | Source | Curves | Points | Purpose |
|---|---|---|---|---|
| **Tier 1: Experimental** | 25 peer-reviewed studies | 105 | 227 | Ground truth — the only data that matters for validation |
| **Tier 2: Calibrated Synthetic** | Generated near real endpoints | 250 | 15,000 | Stability — anchors training near reality |
| **Tier 3: Exploratory Synthetic** | Broader composition/temperature | 700 | 42,000 | Coverage — prevents collapse in unseen regions |

### Data Sources

Experimental data was extracted from publications spanning 2010–2024, covering:
- Fe-(3.9–12.0)Mn alloys with varying C, Al, Si, Mo, Nb
- Annealing temperatures: 500–1000°C
- Processing: cold-rolled, hot-rolled, warm-rolled, LPBF
- Quality: tabulated values, digitized figures, text-reported measurements

Full source list with DOIs: see [data_sources.md](data_sources.md)

### The Hard Part

The hard part was not finding papers. **The hard part was turning heterogeneous, partially incompatible literature evidence into something a kinetics model could learn from without giving fake confidence.**

Studies differ in:
- Alloy systems and compositions
- Initial microstructural states
- Measurement techniques (XRD, EBSD, dilatometry)
- Reporting style (tables vs. figures vs. text)
- Time resolution (2 points vs. 8 points per curve)

---

## Model Architecture

- **Neural ODE** with augmented state (latent_dim=32, augmented_dim=4)
- **Composition encoder** with multi-head attention (4 heads, embed_dim=32)
- **Physics constraints**: monotonicity, bounded output [0,1], thermodynamic features
- **ODE solver**: dopri5 (adaptive step, rtol=0.005, atol=0.0001)
- **Training**: two-stage — Stage 1 on full dataset, Stage 2 fine-tuning on real-only data

---

## Results

### Final Best Model: Stage 2 Fixed (fine-tuned from Stage 1 epoch 109)

| Split | real_rmse | real_mae | real_endpoint_mae | n_real_points |
|---|---|---|---|---|
| **Validation (real)** | 0.2124 | 0.1868 | 0.1868 | 16 curves / 32 pts |
| **Test (real)** | 0.3121 | 0.2670 | 0.2867 | 16 curves / 35 pts |

### Comparison Across Runs

| Run | val_real_rmse | test_real_rmse | Status |
|---|---|---|---|
| Stage 1 (ep 109/120) | 0.21278 | 0.31368 | Completed full training |
| Stage 2 unfixed retry | 0.21266 | 0.32150 | ❌ Test degraded — rejected |
| **Stage 2 fixed retry** | **0.21244** | **0.31214** | ✅ Best on both metrics |

### Honest Interpretation

1. **val_real_rmse ≈ 0.212** — The model learns real experimental trends on average, but this is measured on only 16 validation curves. The score is a rough guide, not a precise guarantee.

2. **test_real_rmse ≈ 0.312** — The test gap (0.312 vs 0.212) reveals a validation-to-test mismatch. This is expected: with only 16 test curves from different studies, a few hard cases can inflate the error significantly.

3. **Stage 2 fine-tuning helped marginally** — The fixed stage 2 improved test_real_rmse by 0.0015 and test_real_mae by 0.004. These are real but small improvements. The bottleneck is not more training — it is more and better data.

4. **The val-test gap is the most important finding** — It tells us that generalization across heterogeneous literature sources is genuinely difficult. A model that fits one lab's data may not fit another's, because the experimental conditions are subtly different.

### What Actually Limits Performance

Based on the full training history and multi-run comparison:

- ❌ **Not the architecture** — the Neural ODE converges well and respects physics constraints
- ❌ **Not the training time** — diminishing returns after epoch ~30 in stage 1
- ✅ **Data heterogeneity** — different labs, different alloys, different measurements
- ✅ **Sparse real curves** — most curves have 2-7 points, limiting temporal learning
- ✅ **Synthetic-real mismatch** — synthetic data helps stability but doesn't fix real-data disagreement
- ✅ **Small holdout sets** — 16 val + 16 test curves make metrics noisy

---

## Training Details

### Stage 1: Full Dataset Training
- 120 epochs on full dataset (1055 curves, 57227 points)
- Best checkpoint at epoch 109 by val_real_rmse
- ~22 hours on Tesla T4 GPU (Kaggle)
- Checkpoint metric: val_real_rmse

### Stage 2: Real-Only Fine-Tuning
- Loaded stage 1 best checkpoint
- Trained on 72 real-only curves (batch_size=1)
- Time-sanitized curves for strict ordering (fixed ODE solver errors)
- 16 epochs before early stopping (patience issue — see below)
- ~8 minutes on Tesla T4

### Early Stopping Note

Stage 2 was cut short at epoch 16/60 due to aggressive early stopping (patience ~10). With only 16 validation curves, random fluctuations trigger the counter. A better setting would be patience ≥ 50 or disabled entirely for a 60-epoch run. Despite this, the epoch-6 checkpoint was already the best.

---

## Project Structure

```
with_kaggle/
├── README.md                   # This file
├── config.py                   # All hyperparameters and paths
├── model.py                    # Neural ODE architecture
├── trainer.py                  # Training loop with two-stage logic
├── real_data.py                # Literature data extraction (25 studies)
├── data_generator.py           # Synthetic data generation
├── features.py                 # Physics-informed feature engineering
├── thermodynamics.py           # CALPHAD-style thermodynamic lookups
├── losses.py                   # Multi-component loss with physics terms
├── optimizer_annealing.py      # Learning rate scheduling
├── explainability.py           # SHAP and sensitivity analysis
├── visualizations.py           # Publication figure generation
├── main.py                     # Entry point
├── data/                       # CALPHAD tables and literature validation
├── data_sources.md             # Full citation list with DOIs
├── models/checkpoints/         # Trained model weights
├── figures/                    # Generated publication figures
├── docs/                       # Q-bank and run documentation
└── kaggle_output_zips/         # All Kaggle run outputs (including failures)
```

---

## Reproducibility

### Requirements
```
torch>=2.0
torchdiffeq
numpy
pandas
scipy
matplotlib
scikit-learn
shap
```

### Running
```bash
# Full training (requires GPU, ~22 hours)
python main.py

# Generate figures from checkpoint
python publication_pipeline.py
```

### Kaggle
The training was executed on Kaggle with Tesla T4 GPU. All notebooks, logs, and result zips are preserved in `kaggle_output_zips/` and `../../../kaggle_safe_archive/`.

---

## Key Lessons Learned

1. **Literature data is abundant but not ML-ready.** Converting published metallurgy into model-grade kinetics data requires significant manual curation and quality judgment.

2. **Synthetic augmentation stabilizes training but doesn't fix generalization.** The model trains smoothly with synthetic support, but real-data performance plateaus because synthetic curves don't capture the true experimental variability.

3. **Normalized metrics can be misleading.** Normalized RMSE looks excellent (~0.024) while real-scale RMSE (~0.21–0.31) reveals the practical limitation. Always report metrics in physical units.

4. **Better optimization ≠ better metallurgical usefulness.** After a certain point, more epochs or fancier training tricks don't help because the bottleneck is data quality and consistency, not model capacity.

5. **The val-test gap is informative, not embarrassing.** It tells us something real about cross-study generalization in metallurgy data.

---

## Citation

If referencing this work:

> Physics-Constrained Neural ODE for Austenite Reversion Kinetics in Medium-Mn Steels: Testing the Limits of Predictive Modeling with Heterogeneous Literature Data. Atin Datta, 2026.

---

## Acknowledgments

Training compute provided by Kaggle (Tesla T4 GPU, 30 hrs/week).
Experimental data extracted from 25 peer-reviewed publications (see [data_sources.md](data_sources.md)).
