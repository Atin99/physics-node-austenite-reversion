# Handoff Prompt — Physics-Constrained Neural ODE Project

## Use this prompt in the next session

---

**Project:** `c:\project 4\project_4\` — Physics-Constrained Latent Neural ODE for Austenite Reversion Kinetics  
**Repo:** https://github.com/Atin99/physics-node-austenite-reversion  
**Conversation:** d170b734-6ae8-4de3-9ac6-7292b0ec54bc

## What was done this session

1. **Streamlit app rewritten** (`src/streamlit_app.py`) — warm neutral design, no emojis, 4 tabs (Prediction, Temperature Sweep, Dataset, Phase Diagram). Works at localhost:8501 via `launch.bat`.

2. **Technical report written** (`docs/TECHNICAL_REPORT.md`) — full academic report with actual numbers.

3. **Deployment files created** — `Dockerfile`, `render.yaml`, `launch.bat`, `launch.ps1`, `.streamlit/config.toml`.

4. **Thermodynamics FIXED** — `src/thermodynamics.py` had broken Ac1 formula (Andrews, overestimated by 50-100C for medium-Mn steels) and wrong f_eq (returned 0 or 1.0 instead of realistic 0.30-0.65). Both recalibrated against the 25-study literature dataset. CALPHAD tables regenerated at 20x20x30.

5. **Model RETRAINED on Kaggle** with fixed thermodynamics. Results:
   - OLD: val_real_rmse=0.212, test_real_rmse=0.312
   - NEW: val_real_rmse=0.161, test_real_rmse=0.131
   - Massive improvement. The new checkpoint is at `kaggle/runs/run_08_retrain_v2/retrain_v2_artifacts/stage2_fixed_best.pt`

6. **New checkpoint ALREADY COPIED** to `models/stage2_fixed_best.pt` (the copy command succeeded).

## What needs to be done NOW

### Step 1: Re-validate with the new checkpoint
```
python validate_model.py
```
Run this and check the results. Should show much better numbers than before.

### Step 2: Commit and push everything
```
git add -A
git commit -m "install retrained model with fixed thermodynamics (val_rmse=0.161, test_rmse=0.131)"
git push
```

### Step 3: Write Project Conclusion Report
Create `docs/PROJECT_CONCLUSION.md` — a final wrap-up document covering:
- What was built, what was learned, what the limits are
- Final metrics table (old vs new)
- Honest assessment of the val-test gap
- What would improve things further

### Step 4: Publication Route Plan
The user wants to publish this, solo, free, even at a low-level venue. Plan:
- Target: Computational Materials Science (Elsevier) or Materials Today Communications (open access option)
- Backup: SSRN preprint, arXiv (cond-mat.mtrl-sci + cs.LG), or Zenodo DOI
- The user does NOT have LaTeX — the paper draft is in markdown at `docs/PAPER_DRAFT.md`
- Need to convert to a journal-ready format
- Remove ALL AI-sounding language. Make it sound like a human researcher wrote it.
- The user is a solo undergrad at Jadavpur University (Metallurgy dept)

### Step 5: Clean up for publication
- Update `README.md` with new metrics
- Update `docs/TECHNICAL_REPORT.md` with new numbers
- Update `docs/VALIDATION_REPORT.md` with new results
- Make sure all language is human, no "leveraging", "harness", "cutting-edge", etc.

### Step 6: Push final state to GitHub

## Key files
- `src/thermodynamics.py` — fixed Ac1/f_eq formulas (the main fix this session)
- `models/stage2_fixed_best.pt` — new best checkpoint (val=0.161, test=0.131)
- `validate_model.py` — backend test script (run to verify)
- `kaggle/runs/run_08_retrain_v2/` — full Kaggle run artifacts
- `docs/PAPER_DRAFT.md` — paper draft (needs de-AI-ification)
- `src/streamlit_app.py` — app (already working)

## User preferences
- No emojis in anything
- No glassmorphism/dark-blue design
- No AI-sounding language — "Reddit-comment, no-fluff" style
- Solo project, no viva defense, stands on its own
- Publication at any level is fine, even low-tier, but must be real
- The user also downloaded `publication-roadmap.html` from Claude — check Downloads for reference
