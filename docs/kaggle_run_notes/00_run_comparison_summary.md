# Four-Run Comparison Summary

## Bottom Line

The four archived runs should not be treated as equally important.

- Use the 50-epoch run only as an early diagnostic reference.
- Use the 150-epoch SWAG run for dataset scale, runtime, and stability context.
- Use the 30-epoch fast run only as proof that the repaired training path executed end to end.
- Use the 60-epoch accelerated run as the main time-bounded completed artifact from this round.

## Run Matrix

| Run | Archive | Main role | Dataset size | Epochs | Runtime | Best logged objective | Recommended use |
|---|---|---|---:|---:|---:|---:|---|
| Diagnostic 50-epoch | `results (2).zip` | Early pipeline check | not fully documented in archive summary | 50 | not retained in final summary | `-1.380907416343689` | Comparison only |
| Long 150-epoch SWAG | `results (3).zip` | Full-scale long run | 2,607 curves / 156,206 points | 150 | 202.0 min | `-1.646647265979222` | Dataset/training procedure context |
| Fast smoke 30-epoch | `results (4).zip` | Code-path validation | 267 curves / 3,182 points | 30 | 0.32 min | `-0.03577897697687149` | Smoke test only |
| Accelerated 60-epoch | `results (5).zip` | Final budget-constrained completed run | 2,607 curves / 124,970 points | 60 | 17.47 min | `-0.5500311085156032` | Main completed artifact from this cycle |

## Important Cautions

1. The logged validation totals are weighted training objectives, not plain prediction-error metrics.
2. The 150-epoch run is useful for coverage and stability, but its stored raw validation component history stayed nearly flat.
3. The 30-epoch run is too small to support any serious final claim.
4. The 60-epoch accelerated run is the most usable completed run under the time constraint, but it is still a reduced-budget training configuration.

## Recommended Project Use

- For data coverage and training-corpus description, cite the 150-epoch SWAG run.
- For a completed, practical training artifact from the repaired fast path, cite the accelerated 60-epoch run.
- Mention the 50-epoch and 30-epoch runs only as intermediate diagnostics.

## Suggested Neutral Wording

> Multiple training passes were retained during development, including an early 50-epoch diagnostic run, a long 150-epoch SWAG run at full data scale, a small smoke-test run used to validate a repaired fast training path, and a final 60-epoch accelerated run completed under remaining compute limits. The long SWAG run is most useful for documenting dataset scale and training stability, while the accelerated 60-epoch run serves as the main completed artifact from the final time-bounded training cycle.
