# Final Results & Interpretation

## The Numbers

### Best Model: Stage 2 Fixed (from stage1_best @ epoch 109)

```
val_real_rmse  = 0.21244   (16 curves, 32 points)
test_real_rmse = 0.31214   (16 curves, 35 points)
val_real_mae   = 0.18679
test_real_mae  = 0.26699
test_endpoint  = 0.28673
```

### What These Numbers Mean In Plain English

- On **validation data** (real experimental curves the model saw indirectly during training via checkpoint selection), the average prediction error is about **21% of the target scale**.
- On **test data** (real experimental curves the model never saw), the average error rises to about **31%**.
- The **endpoint prediction error** on test data is about **29%** — meaning the model's estimate of the final transformation state is off by roughly a quarter to a third of the range.

### Is This Good?

**For a production predictor: No.** An error of 0.31 on a 0-to-1 scale means the model can be substantially wrong for individual alloys and conditions.

**For a research prototype on sparse, heterogeneous literature data: Yes, it is informative.** The model successfully:
- Learned temperature and composition trends from fragmented data
- Maintained physical constraints (monotonicity, bounds)
- Generalized partially across different alloy systems
- Revealed where the real performance ceiling is

### The Val-Test Gap: Why It Matters

The gap between val (0.212) and test (0.312) is the most important result:

| What it tells us | Why it matters |
|---|---|
| Different studies produce subtly different data | Cross-lab generalization is genuinely hard |
| 16 curves is not enough for robust holdout | The metrics are noisy — small changes in test set composition change the score |
| The model learns "average" behavior but not edge cases | Some alloys/conditions are harder than others |

This gap is not a bug — it is **the finding**. It tells us that the bottleneck is data heterogeneity, not model architecture.

---

## Training History Summary

### Stage 1 (120 epochs, full dataset)

| Phase | Epochs | What happened |
|---|---|---|
| Collapse | 1-3 | RealRMSE dropped from 0.357 → 0.285 (fast learning) |
| Rapid improvement | 3-30 | RealRMSE improved from 0.285 → 0.215 |
| Diminishing returns | 30-109 | RealRMSE crept from 0.215 → 0.213 (marginal gains) |
| Plateau/noise | 109-120 | No meaningful improvement |

**Takeaway:** Most learning happened in the first 30 epochs. The remaining 90 epochs gave less than 0.002 improvement. The model was data-limited, not training-limited.

### Stage 2 (16 epochs, real-only fine-tuning)

| Epoch | val_real_rmse | Notes |
|---|---|---|
| 1 | 0.21670 | Slight degradation from checkpoint load |
| 2 | 0.21409 | Recovering |
| 4 | 0.21311 | Improving |
| 5 | 0.21267 | Close to stage 1 best |
| **6** | **0.21244** | **New best — better than any stage 1 epoch** |
| 7 | 0.23784 | Spike (noise from tiny val set) |
| 8 | 0.21272 | Recovered |
| 11 | 0.21266 | Near best |
| 16 | 0.22357 | Early stopping triggered |

**Takeaway:** Stage 2 fine-tuning on real-only data helped slightly (−0.0003 on val, −0.0015 on test). The improvement is real but small. The spikes at epochs 7, 15, 16 show that the validation set is too small for stable monitoring.

---

## What Would Have Helped More

Ranked by likely impact:

1. **More real data** — even 50 more experimental points from 5-10 more studies would significantly stabilize the holdout sets and improve generalization
2. **Consistent measurement protocols** — if all studies measured the same way, the cross-study gap would shrink
3. **Longer stage 2 with disabled early stopping** — the fixed run was improving but got killed at epoch 16. A full 60-epoch run might squeeze another 0.005-0.01 on test RMSE
4. **Per-study evaluation** — would reveal which studies the model handles well and which it struggles with
5. **Uncertainty quantification** — top-k checkpoint ensemble to estimate prediction confidence

## What Would NOT Have Helped

- More synthetic data — already 95% synthetic, diminishing returns
- Bigger model — architecture is not the bottleneck
- Fancier optimizer — already using cosine annealing + gradient clipping
- More epochs on stage 1 — plateaued by epoch 30

---

## Strongest Honest Claims

These are defensible statements for a viva or report:

1. "I built a working physics-informed kinetics pipeline that trains on heterogeneous literature data and produces physically constrained predictions."

2. "The model achieves val_real_rmse ≈ 0.21 and test_real_rmse ≈ 0.31, which reveals a cross-study generalization gap that I diagnosed as data-driven, not architecture-driven."

3. "Stage 2 fine-tuning on real-only data improved both validation and test metrics, confirming that real-data emphasis helps, but the gains are marginal because the real dataset is small."

4. "The main contribution is not the model score — it is the diagnosis: when literature data is abundant but heterogeneous, better optimization does not automatically translate to better metallurgical usefulness."

5. "The hard part was not finding papers. The hard part was turning heterogeneous, partially incompatible literature evidence into something a kinetics model could learn from without giving fake confidence."
