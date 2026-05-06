Project 4.3 HiFi20 Kaggle Run
=============================

Use this package when you want better quality than the fast deadline run while staying inside a 20 hour Kaggle GPU budget.

Expected first training header:

PROJECT 4.3 HIFI20 WRAPPER
Run profile: balanced
Profile params | baseline_n_synth=3500 | stage1_epochs=90 | stage2_epochs=180
Model params | batch=32 amp=1 adjoint=0 sn=0 rtol=0.0002 atol=1e-06 max_steps=3000 time=log10
Train batches: about 39-45, not 114

If Kaggle shows Batch: 16, AMP: False, Adjoint: True, or Train batches: 114, it is running an old bundle.
