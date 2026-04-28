# Part 2: User-Shared 150-Epoch Kaggle Run

Status: nonessential, removable, but useful for report notes.

## Purpose

This file stores the useful parts of the Kaggle console output shared on `2026-04-25`. Even though this run is not strong enough to be the final headline result, it still contains report-usable dataset, training, and stability information.

## Sources

- User-shared Kaggle console output from `2026-04-25`
- Artifact inspected: `C:\Users\dmana\Downloads\results (3).zip`
- Checkpoint inside artifact: `medium_mn_neural_ode/models/checkpoints/physics_node_150ep_swag.pt`

## Experimental Sources Captured In The Shared Log

- `gibbs_2011`: Fe-7.1Mn-0.1C, 5 points, 575-675 C
- `luo_2011`: Fe-5.0Mn-0.2C, 7 points, 650 C
- `pmc11053108_2024`: Fe-4.7Mn-0.16C-1.6Al-0.2Si-0.2Mo, 7 points, 640-1000 C
- `pmc11173901_2024`: Fe-9.4Mn-0.2C-4.3Al, 3 points, 750-850 C
- `pmc6266817_2018`: Fe-5.0Mn-0.12C-1.0Al-0.2Mo-0.05Nb, 6 points, 600-750 C
- `pmc6266817_2018_HR`: Fe-5.0Mn-0.12C-1.0Al-0.2Mo-0.05Nb, 4 points, 600-690 C
- `yan_2022`: Fe-6.0Mn-0.15C-1.5Al-0.3Si, 3 points, 650-710 C
- `aliabad_2026`: Fe-6.0Mn-0.4C-2.0Al-1.0Si, 2 points, 680 C
- `frontiers_2020`: Fe-8.0Mn-0.2C-3.0Al, 2 points, 680 C
- `lpbf_2021`: Fe-3.93Mn-0.23C-2.01Al-0.51Si, 3 points, 25-750 C
- `demoor_2011`: Fe-7.1Mn-0.11C, 5 points, 600 C
- `lee_decooman_2014`: Fe-5.8Mn-0.12C-2.0Al, 6 points, 625-750 C
- `nakada_2014`: Fe-6.0Mn, 7 points, 500-675 C
- `shi_2010`: Fe-5.0Mn-0.2C, 5 points, 620-680 C
- `arlazarov_2012`: Fe-5.0Mn-0.1C, 5 points, 600-700 C
- `han_2014`: Fe-9.0Mn-0.05C, 6 points, 600-725 C
- `miller_2013`: Fe-5.8Mn-0.1C-0.5Si, 5 points, 600-700 C
- `hu_luo_2017`: Fe-7.0Mn-0.1C-0.5Al, 4 points, 600-675 C
- `cai_2016`: Fe-10.0Mn-0.3C-2.0Al, 4 points, 625-700 C
- `zhao_2014`: Fe-7.9Mn-0.07C-0.05Al-0.14Si, 7 points, 580-700 C
- `suh_2017`: Fe-6.0Mn-0.1C-3.0Al, 5 points, 700-800 C
- `sun_2018`: Fe-12.0Mn-0.05C, 8 points, 575-675 C
- `ma_2019`: Fe-8.0Mn-0.2C-2.0Al-0.5Si, 6 points, 625-750 C
- `demoor_2015`: Fe-7.0Mn-0.1C, 6 points, 620 C
- `hausman_2017`: Fe-6.0Mn-0.3C-0.5Si, 4 points, 575-650 C

## Keepable Facts For The Report

- Total experimental data loaded: `125` points from `25` studies
- User CSV files found at runtime: `none`
- Real measurements converted into `107` training curves with `6206` total points
- Calibrated synthetic curves: `500` curves with `30000` points
- Exploratory synthetic curves: `2000` curves with `120000` points
- Full dataset: `2607` curves and `156206` total points
- Training device: `cuda`
- Training epochs: `150`
- Batch size: `128`
- Learning rate at start: `0.0003`
- AMP: `False`
- Adjoint mode: `True`
- SWAG enabled: `True`
- SWAG checkpoint saved: `physics_node_150ep_swag.pt`
- Training time reported in log: `202.0` minutes
- Logged monotonicity violations: `0.00%` throughout the run

## Useful But Not Safe As Final Headline Claims

- Best total validation objective in checkpoint: `-1.646647265979222`
- Train total objective moved from `-0.005699937872122973` to `-1.7724229778562273`
- Validation total objective moved from `-0.01639623966600214` to `-1.646647265979222`

These totals come from the weighted homoscedastic objective, so they are useful as optimization logs but not strong standalone evidence of final predictive quality.

## Extra Caution From Checkpoint Inspection

The stored raw validation component history in the checkpoint remained flat:

- `val_data = 0.0003581413662426972`
- `val_physics = 8.349732552354843e-07`
- `val_mono = 4.256717061374664e-08`
- `val_bound = 3.962435131839254e-15`

This is why the run can be cited for data coverage, training stability, and artifact generation, but should not be the main evidence for model performance.

## Suggested Report Sentence

> The final training corpus comprised 2,607 kinetic curves (156,206 total points), including 107 curves derived from 125 experimental measurements across 25 published studies, supplemented by 500 calibrated synthetic and 2,000 exploratory synthetic curves. A 150-epoch CUDA training run with SWAG completed successfully in 202 minutes and produced a final uncertainty-aware checkpoint without monotonicity violations.

## Recommended Use

Use this file in:

- Data or Materials and Methods
- Training Procedure
- Implementation Details
- Limitations or methodological caution

Avoid using it as the sole basis for a final accuracy/performance claim.
