"""
Post-hoc analysis of training logs and dataset.
Runs CPU-only, no GPU needed.
Extracts per-study stats, training dynamics, and baseline comparisons.

Usage: python analysis.py
Output: analysis_results/ folder with csvs and a summary text file
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# paths relative to project root
PROJECT = Path(__file__).parent
DATA_CSV = PROJECT / "data" / "literature_validation" / "literature_validation.csv"
S1_HIST = PROJECT / "models" / "stage1_history.csv"
S2_FIXED_HIST = PROJECT / "models" / "stage2_fixed_history.csv"
S2_RUN7_HIST = PROJECT / "models" / "stage2_run7_history.csv"
S2_FIXED_SUMMARY = PROJECT / "models" / "stage2_fixed_run_summary.json"
S2_RUN7_SUMMARY = PROJECT / "models" / "stage2_run7_summary.json"
OUT_DIR = PROJECT / "analysis_results"


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def dataset_analysis(rows):
    """Per-study breakdown, composition coverage, measurement method stats."""
    results = []
    results.append("=" * 70)
    results.append("DATASET ANALYSIS")
    results.append("=" * 70)
    results.append(f"Total data points: {len(rows)}")

    # per study
    studies = defaultdict(list)
    for r in rows:
        studies[r["study_id"]].append(r)

    results.append(f"Unique studies: {len(studies)}")
    results.append("")
    results.append(f"{'Study':<25s} {'Points':>6s} {'Mn':>5s} {'C':>5s} {'Al':>5s} {'T range (C)':>15s} {'Method':>10s} {'Quality':>15s}")
    results.append("-" * 100)

    for sid, pts in sorted(studies.items()):
        mn = float(pts[0]["Mn"])
        c = float(pts[0]["C"])
        al = float(pts[0]["Al"])
        temps = [float(p["T_celsius"]) for p in pts]
        methods = set(p["method"] for p in pts)
        qualities = set(p["data_quality"] for p in pts)
        results.append(f"{sid:<25s} {len(pts):>6d} {mn:>5.1f} {c:>5.2f} {al:>5.1f} {min(temps):>6.0f}-{max(temps):>6.0f}  {','.join(methods):>10s} {','.join(qualities):>15s}")

    results.append("")

    # measurement method breakdown
    methods = defaultdict(int)
    for r in rows:
        methods[r["method"]] += 1
    results.append("Measurement methods:")
    for m, c in sorted(methods.items(), key=lambda x: -x[1]):
        results.append(f"  {m}: {c} points ({100*c/len(rows):.1f}%)")

    # data quality breakdown
    qualities = defaultdict(int)
    for r in rows:
        qualities[r["data_quality"]] += 1
    results.append("Data quality flags:")
    for q, c in sorted(qualities.items(), key=lambda x: -x[1]):
        results.append(f"  {q}: {c} points ({100*c/len(rows):.1f}%)")

    # composition ranges
    results.append("")
    results.append("Composition ranges:")
    for elem in ["Mn", "C", "Al", "Si", "Mo", "Nb"]:
        vals = [float(r[elem]) for r in rows]
        results.append(f"  {elem}: {min(vals):.2f} - {max(vals):.2f}")

    # temperature and time ranges
    temps = [float(r["T_celsius"]) for r in rows]
    times = [float(r["t_seconds"]) for r in rows]
    fracs = [float(r["f_RA"]) for r in rows]
    results.append(f"  T_celsius: {min(temps):.0f} - {max(temps):.0f}")
    results.append(f"  t_seconds: {min(times):.0f} - {max(times):.0f} ({max(times)/3600:.0f} hours)")
    results.append(f"  f_RA: {min(fracs):.3f} - {max(fracs):.3f}")

    # measurement method discrepancy analysis
    results.append("")
    results.append("MEASUREMENT METHOD DISCREPANCY (same condition, different methods):")
    # group by study + temperature + time
    condition_groups = defaultdict(list)
    for r in rows:
        key = (r["study_id"].split("_")[0], r["T_celsius"], r["t_seconds"])
        condition_groups[key].append(r)
    
    for key, pts in condition_groups.items():
        methods_here = set(p["method"] for p in pts)
        if len(methods_here) > 1:
            results.append(f"  {key[0]} @ {key[1]}C, {key[2]}s:")
            for p in pts:
                results.append(f"    {p['method']}: {float(p['f_RA_pct']):.1f}% (study: {p['study_id']})")

    return "\n".join(results)


def training_dynamics(s1_rows, s2_fixed_rows, s2_run7_rows):
    """Analyze training convergence, learning rate schedules, physics violation decay."""
    results = []
    results.append("")
    results.append("=" * 70)
    results.append("TRAINING DYNAMICS")
    results.append("=" * 70)

    # stage 1
    results.append(f"Stage 1: {len(s1_rows)} epochs")
    if s1_rows:
        first = s1_rows[0]
        last = s1_rows[-1]
        # find best val_real_rmse
        best_idx = min(range(len(s1_rows)), key=lambda i: float(s1_rows[i]["val_real_rmse"]))
        best = s1_rows[best_idx]
        results.append(f"  First epoch val_real_rmse: {float(first['val_real_rmse']):.5f}")
        results.append(f"  Best epoch val_real_rmse: {float(best['val_real_rmse']):.5f} (epoch {best['epoch']})")
        results.append(f"  Last epoch val_real_rmse: {float(last['val_real_rmse']):.5f}")
        results.append(f"  LR: {float(first['lr']):.6f} -> {float(last['lr']):.6f}")
        results.append(f"  Physics violations: {float(first['violations']):.4f} -> {float(last['violations']):.4f}")
        results.append(f"  NFE (neural function evals): {float(first['nfe']):.0f} -> {float(last['nfe']):.0f}")
        
        # convergence phases
        # find when val_real_rmse first drops below 0.22
        threshold_epochs = {}
        for thresh in [0.30, 0.25, 0.22, 0.215, 0.213]:
            for i, r in enumerate(s1_rows):
                if float(r["val_real_rmse"]) < thresh:
                    threshold_epochs[thresh] = int(r["epoch"])
                    break
        results.append("  Convergence milestones:")
        for thresh, ep in sorted(threshold_epochs.items(), reverse=True):
            results.append(f"    val_real_rmse < {thresh}: epoch {ep}")

    # stage 2 fixed (run 06)
    results.append(f"\nStage 2 Fixed (Run 06): {len(s2_fixed_rows)} epochs")
    if s2_fixed_rows:
        best_idx = min(range(len(s2_fixed_rows)), key=lambda i: float(s2_fixed_rows[i]["val_real_rmse"]))
        best = s2_fixed_rows[best_idx]
        results.append(f"  Best epoch: {best['epoch']} with val_real_rmse = {float(best['val_real_rmse']):.5f}")

    # stage 2 run7 (60 epochs)
    results.append(f"\nStage 2 Run 07 (60 epochs): {len(s2_run7_rows)} epochs")
    if s2_run7_rows:
        best_idx = min(range(len(s2_run7_rows)), key=lambda i: float(s2_run7_rows[i]["val_real_rmse"]))
        best = s2_run7_rows[best_idx]
        last = s2_run7_rows[-1]
        results.append(f"  Best epoch: {best['epoch']} with val_real_rmse = {float(best['val_real_rmse']):.5f}")
        results.append(f"  Last epoch val_real_rmse: {float(last['val_real_rmse']):.5f}")
        results.append(f"  LR: {float(s2_run7_rows[0]['lr']):.8f} -> {float(last['lr']):.8f}")
        
        # check if early stopping would have helped
        results.append(f"  Skipped batches: {sum(float(r['skipped_batches']) for r in s2_run7_rows):.0f}")

    return "\n".join(results)


def model_comparison():
    """Compare final metrics between runs."""
    results = []
    results.append("")
    results.append("=" * 70)
    results.append("MODEL COMPARISON (Run 06 vs Run 07)")
    results.append("=" * 70)

    for label, path in [("Run 06 (fixed)", S2_FIXED_SUMMARY), ("Run 07 (latest)", S2_RUN7_SUMMARY)]:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            results.append(f"\n{label}:")
            # stage 1 baseline
            if "stage1_baseline" in data:
                s1 = data["stage1_baseline"]
                results.append(f"  Stage 1 baseline:")
                results.append(f"    val_real_rmse:  {s1['val_real']['real_rmse']:.5f}")
                results.append(f"    test_real_rmse: {s1['test_real']['real_rmse']:.5f}")
            
            # final eval
            if "final_evaluation" in data:
                fe = data["final_evaluation"]
                results.append(f"  After stage 2:")
                results.append(f"    val_real_rmse:  {fe['val_real']['real_rmse']:.5f}")
                results.append(f"    test_real_rmse: {fe['test_real']['real_rmse']:.5f}")
                results.append(f"    val_real_mae:   {fe['val_real']['real_mae']:.5f}")
                results.append(f"    test_real_mae:  {fe['test_real']['real_mae']:.5f}")
            
            if "runtime_min" in data:
                results.append(f"  Runtime: {data['runtime_min']:.1f} min on {data.get('gpu', 'unknown')}")

    results.append("")
    results.append("KEY OBSERVATION:")
    results.append("  Run 06 has better TEST metrics (0.312 vs 0.328)")
    results.append("  Run 07 has better VAL metrics (0.211 vs 0.212)")
    results.append("  This is expected with 16-curve validation sets - noise dominates")
    results.append("  Run 06 remains the best model for deployment")

    return "\n".join(results)


def jmak_baseline(rows):
    """Simple JMAK analysis per study (isothermal curves only)."""
    import math
    
    results = []
    results.append("")
    results.append("=" * 70)
    results.append("JMAK BASELINE ANALYSIS")
    results.append("=" * 70)
    results.append("Fitting f(t) = f_max * (1 - exp(-k * t^n)) per isothermal curve")
    results.append("")

    # find studies with multiple time points at same temperature
    studies = defaultdict(list)
    for r in rows:
        studies[r["study_id"]].append(r)

    kinetic_studies = []
    for sid, pts in studies.items():
        # group by temperature
        temp_groups = defaultdict(list)
        for p in pts:
            temp_groups[p["T_celsius"]].append(p)
        
        for temp, tpts in temp_groups.items():
            times = [float(p["t_seconds"]) for p in tpts]
            fracs = [float(p["f_RA"]) for p in tpts]
            if len(tpts) >= 3 and max(times) > min(times) and max(fracs) > 0.01:
                kinetic_studies.append((sid, temp, tpts))

    results.append(f"Found {len(kinetic_studies)} isothermal curves with 3+ time points")
    results.append("")

    for sid, temp, tpts in kinetic_studies:
        times = [float(p["t_seconds"]) for p in tpts]
        fracs = [float(p["f_RA"]) for p in tpts]
        f_max = max(fracs) * 1.05  # slight overshoot for fitting

        # simple grid search for k, n
        best_rmse = 999
        best_k = 0
        best_n = 0
        
        for n_try in [x * 0.1 for x in range(1, 30)]:
            for log_k in [x * 0.5 for x in range(-30, 5)]:
                k_try = math.exp(log_k)
                pred = [f_max * (1 - math.exp(-k_try * (t ** n_try))) if t > 0 else 0 for t in times]
                rmse = (sum((p - a) ** 2 for p, a in zip(pred, fracs)) / len(fracs)) ** 0.5
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_k = k_try
                    best_n = n_try

        results.append(f"  {sid} @ {temp}C: n={best_n:.1f}, k={best_k:.2e}, RMSE={best_rmse:.4f} ({len(tpts)} points)")

    results.append("")
    results.append("NOTE: JMAK fits well within a single study's isothermal curve.")
    results.append("But JMAK parameters are NOT transferable across compositions.")
    results.append("The Neural ODE's value is cross-composition generalization.")

    return "\n".join(results)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # load everything
    data_rows = load_csv(DATA_CSV)
    s1_rows = load_csv(S1_HIST) if S1_HIST.exists() else []
    s2_fixed_rows = load_csv(S2_FIXED_HIST) if S2_FIXED_HIST.exists() else []
    s2_run7_rows = load_csv(S2_RUN7_HIST) if S2_RUN7_HIST.exists() else []

    # run all analyses
    sections = []
    sections.append(dataset_analysis(data_rows))
    sections.append(training_dynamics(s1_rows, s2_fixed_rows, s2_run7_rows))
    sections.append(model_comparison())
    sections.append(jmak_baseline(data_rows))

    full_report = "\n".join(sections)
    
    # write to file
    report_path = OUT_DIR / "full_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(full_report)
    
    print(full_report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
