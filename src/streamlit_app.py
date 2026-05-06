"""
Multi-Pipeline Streamlit App for Austenite Reversion Kinetics.
Supports: Neural ODE (low-data), Neural ODE (HiFi), kNN, Ridge, Fusion baselines.
Run with: streamlit run src/streamlit_app.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:
    print("streamlit not installed. pip install streamlit")
    sys.exit(1)

import torch
from config import get_config
from thermodynamics import get_Ac1_Ac3, get_equilibrium_RA, get_driving_force
from features import compute_Md30, compute_diffusivity, compute_hollomon_jaffe
from model import PhysicsNODE

st.set_page_config(page_title="PhysicsNODE - Austenite Reversion", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans Pro', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1100px; }
h1 { color: #2c3e50; font-weight: 600; font-size: 1.8rem; }
h2 { color: #34495e; font-weight: 600; font-size: 1.3rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
h3 { color: #4a5568; font-weight: 600; font-size: 1.1rem; }
.metric-box { background: #f7f7f5; border: 1px solid #e2e0dc; border-radius: 6px; padding: 1rem; text-align: center; margin-bottom: 0.5rem; }
.metric-val { font-size: 1.6rem; font-weight: 600; color: #2c3e50; }
.metric-label { font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; }
.info-block { background: #fafaf8; border-left: 3px solid #b8a88a; padding: 0.8rem 1rem; margin: 1rem 0; font-size: 0.9rem; color: #4a5568; }
.pipeline-tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.tag-node { background: #e8f5e9; color: #2e7d32; }
.tag-ml { background: #e3f2fd; color: #1565c0; }
footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)


# ── Pipeline definitions ──
PIPELINES = {
    "Neural ODE (125 pts, best)": {
        "type": "node", "data": "125 real", "tag_class": "tag-node",
        "desc": "Physics-constrained Latent Neural ODE. Best model: test R²=+0.378.",
        "ckpt_names": ["v4_stage2_fixed_best.pt", "stage2_fixed_best.pt"],
        "cfg_override": {},  # uses default v4 config
    },
    "Neural ODE HiFi (125 pts)": {
        "type": "node", "data": "125 real", "tag_class": "tag-node",
        "desc": "Alternate HiFi config (no spectral norm, log10 time). test R²=-3.56.",
        "ckpt_names": ["v43_hifi_stage2_best.pt"],
        "cfg_override": {"hidden_dims": [128, 128, 96], "augmented_dim": 3, "use_spectral_norm": False},
    },
    "Neural ODE Trial2 (125 pts)": {
        "type": "node", "data": "125 real", "tag_class": "tag-node",
        "desc": "Fast-trained v4.3 trial run (run1_trial2). Smaller architecture.",
        "ckpt_names": ["v43_trial2_stage2_best.pt"],
        "cfg_override": {"hidden_dims": [96, 96, 64], "augmented_dim": 2, "use_spectral_norm": False},
    },
    "Neural ODE Final (125 pts)": {
        "type": "node", "data": "125 real", "tag_class": "tag-node",
        "desc": "v4.3 final trial run (run2_final_trial). Fixed + extended checkpoints.",
        "ckpt_names": ["v43_final_stage2_fixed_best.pt", "v43_final_stage2_extended_best.pt"],
        "cfg_override": {"hidden_dims": [96, 96, 64], "augmented_dim": 2, "use_spectral_norm": False},
    },
    "kNN Baseline (373 pts)": {
        "type": "ml", "data": "373 real+transfer", "tag_class": "tag-ml",
        "desc": "k-Nearest Neighbors on expanded dataset. Simple but effective: R²≈0.36.",
        "ml_model": "knn",
    },
    "Ridge Baseline (373 pts)": {
        "type": "ml", "data": "373 real+transfer", "tag_class": "tag-ml",
        "desc": "Ridge regression with polynomial features. R²≈0.35.",
        "ml_model": "ridge",
    },
    "Fusion Ensemble (373 pts)": {
        "type": "ml", "data": "373 real+transfer", "tag_class": "tag-ml",
        "desc": "Weighted average of kNN + Ridge. Best baseline: R²≈0.40.",
        "ml_model": "fusion",
    },
}


# ── Model loaders ──
@st.cache_resource
def load_node_model(ckpt_names, cfg_override_json=None):
    import json, copy
    cfg = get_config()
    model_cfg = copy.deepcopy(cfg.model)
    # Apply architecture overrides for v4.3 checkpoints
    if cfg_override_json:
        overrides = json.loads(cfg_override_json)
        for k, v in overrides.items():
            setattr(model_cfg, k, v)
    model = PhysicsNODE(model_cfg)
    search = []
    for name in ckpt_names:
        search.extend([REPO_ROOT / "models" / name, PROJECT_ROOT / "models" / name])
    loaded_name = None
    for p in search:
        if p.exists():
            ckpt = torch.load(p, map_location='cpu', weights_only=False)
            state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
            model.load_state_dict(state, strict=False)
            loaded_name = p.name
            break
    model.eval()
    return model, cfg, loaded_name


@st.cache_resource
def load_ml_dataset():
    candidates = [
        REPO_ROOT / "data" / "processed" / "ml_ready_with_transfer.csv",
        REPO_ROOT / "data" / "processed" / "ml_ready_real_pool.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            return df, p.name
    return None, None


@st.cache_resource
def train_ml_model(model_type):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import Ridge as RidgeReg
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.pipeline import Pipeline

    df, _ = load_ml_dataset()
    if df is None:
        return None, None

    # Map column names — CSV uses Mn_wt/C_wt/T_anneal_C/Time_sec/RA_fraction
    col_map = {
        'Mn_wt': 'Mn', 'C_wt': 'C', 'Al_wt': 'Al', 'Si_wt': 'Si',
        'T_anneal_C': 'T_celsius', 'Time_sec': 't_seconds',
        'RA_fraction': 'f_RA', 'RA_pct': 'f_RA_pct',
    }
    df = df.rename(columns=col_map)

    feat_cols = [c for c in ['Mn','C','Al','Si','T_celsius','t_seconds'] if c in df.columns]
    target = 'f_RA' if 'f_RA' in df.columns else ('f_RA_pct' if 'f_RA_pct' in df.columns else 'RA_fraction')
    X = df[feat_cols].values
    y = df[target].values
    if y.max() > 1.5:
        y = y / 100.0

    if model_type == "knn":
        pipe = Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor(n_neighbors=5, weights='distance'))])
    elif model_type == "ridge":
        pipe = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(2, include_bias=False)), ('model', RidgeReg(alpha=1.0))])
    elif model_type == "fusion":
        knn = Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor(n_neighbors=5, weights='distance'))])
        ridge = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(2, include_bias=False)), ('model', RidgeReg(alpha=1.0))])
        knn.fit(X, y)
        ridge.fit(X, y)
        return (knn, ridge), feat_cols
    else:
        return None, None

    pipe.fit(X, y)
    return pipe, feat_cols


@st.cache_data
def load_lit_dataset():
    candidates = [
        REPO_ROOT / "data" / "literature" / "literature_validation.csv",
        REPO_ROOT / "data" / "literature_validation" / "literature_validation.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


# ── Prediction functions ──
def predict_node_curve(model, cfg, comp, T_c, t_max_sec):
    T_K = T_c + 273.15
    f_eq, _ = get_equilibrium_RA(comp, T_c, force_fallback=True)
    dG = get_driving_force(comp, T_c, force_fallback=True)
    D = compute_diffusivity(T_K)
    P = compute_hollomon_jaffe(T_K, max(t_max_sec / 2, 1.0))
    static = torch.tensor([[(T_K - cfg.data.T_ref) / cfg.data.T_scale,
        comp['Mn'], comp['C'], comp.get('Al', 0), comp.get('Si', 0),
        np.log10(D + 1e-30), dG / 1000.0, P / 20000.0]], dtype=torch.float32)
    f_eq_t = torch.tensor([[f_eq]], dtype=torch.float32)
    dG_t = torch.tensor([[dG / 1000.0]], dtype=torch.float32)
    t_span = torch.linspace(0, float(t_max_sec), 50)
    mean, lo, hi = model.predict_with_uncertainty(static, f_eq_t, dG_t, t_span, 30)
    Ac1, Ac3 = get_Ac1_Ac3(comp)
    return {
        't_hours': t_span.numpy() / 3600, 'f_RA_mean': mean[0].numpy(),
        'f_RA_lower': lo[0].numpy(), 'f_RA_upper': hi[0].numpy(),
        'f_eq': f_eq, 'Md30': compute_Md30(comp), 'D_Mn': D,
        'Ac1': Ac1, 'Ac3': Ac3, 'delta_G': dG,
    }


def predict_ml_curve(ml_model, feat_cols, comp, T_c, t_max_sec, model_type):
    times = np.linspace(1, t_max_sec, 50)
    preds = []
    for t in times:
        row = {'Mn': comp['Mn'], 'C': comp['C'], 'Al': comp.get('Al', 0),
               'Si': comp.get('Si', 0), 'T_celsius': T_c, 't_seconds': t}
        X = np.array([[row[c] for c in feat_cols]])
        if model_type == "fusion":
            knn_model, ridge_model = ml_model
            p = 0.5 * knn_model.predict(X)[0] + 0.5 * ridge_model.predict(X)[0]
        else:
            p = ml_model.predict(X)[0]
        preds.append(float(np.clip(p, 0, 1)))
    f_eq, _ = get_equilibrium_RA(comp, T_c, force_fallback=True)
    Ac1, Ac3 = get_Ac1_Ac3(comp)
    return {
        't_hours': times / 3600, 'f_RA_mean': np.array(preds),
        'f_RA_lower': np.array(preds) * 0.85, 'f_RA_upper': np.array(preds) * 1.15,
        'f_eq': f_eq, 'Md30': compute_Md30(comp), 'D_Mn': compute_diffusivity(T_c + 273.15),
        'Ac1': Ac1, 'Ac3': Ac3, 'delta_G': get_driving_force(comp, T_c, force_fallback=True),
    }


def render_metric(label, value):
    st.markdown(f"""<div class="metric-box"><div class="metric-val">{value}</div><div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)


# ── MAIN ──
st.title("PhysicsNODE")
st.markdown("Multi-pipeline prediction of austenite reversion kinetics in medium-Mn steels")

df_lit = load_lit_dataset()

# ── Sidebar ──
with st.sidebar:
    st.markdown("### Pipeline Selection")
    pipeline_name = st.selectbox("Choose model", list(PIPELINES.keys()), index=0)
    pipe_info = PIPELINES[pipeline_name]
    tag_cls = pipe_info["tag_class"]
    st.markdown(f'<span class="pipeline-tag {tag_cls}">{pipe_info["data"]}</span>', unsafe_allow_html=True)
    st.markdown(pipe_info["desc"])

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    Predicts retained austenite fraction during intercritical annealing
    of medium-Mn steels (3-12 wt% Mn).

    Trained on 125–373 experimental measurements from 40+ sources
    (published studies, patents, university lab data, industrial reports).

    **Best Neural ODE (test set):**
    - Test R² = **+0.378**
    - Test RMSE: 0.135
    - 12/21 studies positive R²
    """)

# Load selected pipeline
if pipe_info["type"] == "node":
    import json
    override_json = json.dumps(pipe_info.get("cfg_override", {})) or None
    node_model, cfg, loaded_name = load_node_model(tuple(pipe_info["ckpt_names"]), override_json if override_json != '{}' else None)
    if loaded_name:
        st.sidebar.success(f"Loaded: {loaded_name}")
    else:
        st.sidebar.error("No checkpoint found!")
else:
    ml_model, feat_cols = train_ml_model(pipe_info["ml_model"])
    ml_df, ml_src = load_ml_dataset()
    if ml_model:
        st.sidebar.success(f"Trained on: {ml_src}")
    else:
        st.sidebar.error("Training data not found!")

# ── Tabs ──
tab_predict, tab_sweep, tab_data, tab_phase = st.tabs(["Prediction", "Temperature Sweep", "Dataset", "Phase Diagram"])

with tab_predict:
    st.header("Forward Prediction")
    st.markdown(f"Using: **{pipeline_name}**")
    col_input, col_output = st.columns([1, 2])

    with col_input:
        st.subheader("Composition (wt%)")
        Mn = st.slider("Mn", 4.0, 12.0, 7.0, 0.1, key="p_mn")
        C = st.slider("C", 0.05, 0.30, 0.10, 0.01, key="p_c")
        Al = st.slider("Al", 0.0, 3.0, 0.0, 0.1, key="p_al")
        Si = st.slider("Si", 0.0, 2.0, 0.0, 0.1, key="p_si")
        st.subheader("Annealing")
        T = st.slider("Temperature (°C)", 550, 800, 650, 5, key="p_t")
        t_min = st.slider("Time (minutes)", 1, 300, 60, 1, key="p_time")
        run = st.button("Run prediction", key="p_run", type="primary")

    with col_output:
        if run:
            comp = {'Mn': Mn, 'C': C, 'Al': Al, 'Si': Si}
            with st.spinner("Computing..."):
                if pipe_info["type"] == "node":
                    r = predict_node_curve(node_model, cfg, comp, T, t_min * 60)
                else:
                    r = predict_ml_curve(ml_model, feat_cols, comp, T, t_min * 60, pipe_info["ml_model"])

            c1, c2, c3, c4 = st.columns(4)
            with c1: render_metric("Final RA", f"{r['f_RA_mean'][-1]:.1%}")
            with c2: render_metric("Equilibrium", f"{r['f_eq']:.1%}")
            with c3: render_metric("Ac1 / Ac3", f"{r['Ac1']:.0f} / {r['Ac3']:.0f} °C")
            with c4: render_metric("Md30", f"{r['Md30']:.0f} °C")

            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams.update({'font.size': 10, 'font.family': 'serif'})
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.fill_between(r['t_hours'], r['f_RA_lower'] * 100, r['f_RA_upper'] * 100,
                            alpha=0.2, color='#5b7553', label='95% CI')
            ax.plot(r['t_hours'], r['f_RA_mean'] * 100, '-', color='#5b7553', lw=2, label='Prediction')
            ax.axhline(r['f_eq'] * 100, color='#b8860b', ls='--', alpha=0.6, label=f'Equilibrium ({r["f_eq"]:.1%})')
            ax.set_xlabel('Time (hours)'); ax.set_ylabel('Retained austenite (%)')
            ax.set_title(f'Fe-{Mn:.1f}Mn-{C:.2f}C-{Al:.1f}Al at {T}°C — {pipeline_name}', fontsize=11)
            ax.legend(fontsize=9, framealpha=0.8); ax.set_ylim(bottom=0); ax.grid(True, alpha=0.3)
            fig.tight_layout(); st.pyplot(fig); plt.close()

            with st.expander("Thermodynamic details"):
                st.markdown(f"""
                - Driving force (ΔG): {r['delta_G']:.0f} J/mol
                - Mn diffusivity at {T}°C: {r['D_Mn']:.2e} m²/s
                - Pipeline: {pipeline_name} ({pipe_info['data']})
                """)

with tab_sweep:
    st.header("Temperature Sweep")
    st.markdown(f"Using: **{pipeline_name}**")
    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.subheader("Composition (wt%)")
        Mn2 = st.slider("Mn", 4.0, 12.0, 7.0, 0.1, key="s_mn")
        C2 = st.slider("C", 0.05, 0.30, 0.10, 0.01, key="s_c")
        Al2 = st.slider("Al", 0.0, 3.0, 0.0, 0.1, key="s_al")
        Si2 = st.slider("Si", 0.0, 2.0, 0.0, 0.1, key="s_si")
        t_hold = st.slider("Hold time (minutes)", 10, 300, 60, 10, key="s_time")
        run2 = st.button("Run sweep", key="s_run", type="primary")

    with col_out:
        if run2:
            comp2 = {'Mn': Mn2, 'C': C2, 'Al': Al2, 'Si': Si2}
            Ac1, Ac3 = get_Ac1_Ac3(comp2)
            T_range = np.linspace(max(Ac1 - 20, 550), min(Ac3 + 20, 800), 20)
            results = []
            progress = st.progress(0)
            for i, Ti in enumerate(T_range):
                try:
                    if pipe_info["type"] == "node":
                        ri = predict_node_curve(node_model, cfg, comp2, float(Ti), t_hold * 60)
                    else:
                        ri = predict_ml_curve(ml_model, feat_cols, comp2, float(Ti), t_hold * 60, pipe_info["ml_model"])
                    results.append({'T': Ti, 'RA': ri['f_RA_mean'][-1] * 100, 'f_eq': ri['f_eq'] * 100})
                except Exception:
                    pass
                progress.progress((i + 1) / len(T_range))
            progress.empty()
            if results:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.rcParams.update({'font.size': 10, 'font.family': 'serif'})
                fig, ax = plt.subplots(figsize=(7, 4))
                temps = [r['T'] for r in results]; ras = [r['RA'] for r in results]; feqs = [r['f_eq'] for r in results]
                ax.plot(temps, ras, 'o-', color='#5b7553', lw=2, markersize=5, label='Predicted RA')
                ax.plot(temps, feqs, '--', color='#b8860b', alpha=0.6, label='Equilibrium')
                ax.axvline(Ac1, color='#cc4444', ls=':', alpha=0.5, label=f'Ac1 = {Ac1:.0f}°C')
                ax.axvline(Ac3, color='#4444cc', ls=':', alpha=0.5, label=f'Ac3 = {Ac3:.0f}°C')
                ax.set_xlabel('Temperature (°C)'); ax.set_ylabel('Retained austenite (%)')
                ax.set_title(f'Fe-{Mn2:.1f}Mn-{C2:.2f}C at {t_hold} min — {pipeline_name}', fontsize=11)
                ax.legend(fontsize=9, framealpha=0.8); ax.set_ylim(bottom=0); ax.grid(True, alpha=0.3)
                fig.tight_layout(); st.pyplot(fig); plt.close()
                best = max(results, key=lambda x: x['RA'])
                st.markdown(f'<div class="info-block">Peak RA: <b>{best["RA"]:.1f}%</b> at <b>{best["T"]:.0f}°C</b></div>', unsafe_allow_html=True)

with tab_data:
    st.header("Literature Dataset")
    if df_lit is not None:
        st.markdown(f"125 experimental measurements from 25 published studies (2010-2024).")
        c1, c2, c3 = st.columns(3)
        with c1: render_metric("Data Points", str(len(df_lit)))
        with c2: render_metric("Studies", str(df_lit['study_id'].nunique()))
        with c3: render_metric("Alloys", str(len(df_lit.groupby(['Mn', 'C', 'Al']))))
        st.subheader("Filter")
        fc1, fc2 = st.columns(2)
        with fc1: sel_study = st.multiselect("Study", sorted(df_lit['study_id'].unique()), key="d_study")
        with fc2: sel_method = st.multiselect("Method", sorted(df_lit['method'].unique()), key="d_method")
        filtered = df_lit.copy()
        if sel_study: filtered = filtered[filtered['study_id'].isin(sel_study)]
        if sel_method: filtered = filtered[filtered['method'].isin(sel_method)]
        display_cols = [c for c in ['study_id','Mn','C','Al','T_celsius','t_seconds','f_RA_pct','method','data_quality','doi'] if c in filtered.columns]
        st.dataframe(filtered[display_cols], height=400, use_container_width=True)
        st.subheader("Data Distribution")
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams.update({'font.size': 10, 'font.family': 'serif'})
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sc = axes[0].scatter(filtered['T_celsius'], filtered['f_RA_pct'], c=filtered['Mn'], cmap='YlOrBr', s=30, alpha=0.7, edgecolors='#555', linewidth=0.3)
        axes[0].set_xlabel('Temperature (°C)'); axes[0].set_ylabel('RA (%)'); axes[0].set_title('RA vs Temperature')
        plt.colorbar(sc, ax=axes[0], label='Mn (wt%)')
        sc2 = axes[1].scatter(filtered['Mn'], filtered['f_RA_pct'], c=filtered['T_celsius'], cmap='YlOrBr', s=30, alpha=0.7, edgecolors='#555', linewidth=0.3)
        axes[1].set_xlabel('Mn (wt%)'); axes[1].set_ylabel('RA (%)'); axes[1].set_title('RA vs Mn Content')
        plt.colorbar(sc2, ax=axes[1], label='T (°C)')
        fig.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.error("Dataset CSV not found.")

with tab_phase:
    st.header("Pseudo Phase Diagram")
    st.markdown("Ac1-Ac3 boundaries vs Mn content from empirical correlations.")
    pd_c = st.slider("Carbon content (wt%)", 0.05, 0.30, 0.10, 0.01, key="pd_c")
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({'font.size': 10, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(8, 5))
    Mn_range = np.linspace(4, 12, 60)
    Ac1s, Ac3s = zip(*[get_Ac1_Ac3({'Mn': m, 'C': pd_c}) for m in Mn_range])
    Ac1s, Ac3s = np.array(Ac1s), np.array(Ac3s)
    ax.fill_between(Mn_range, Ac1s, Ac3s, alpha=0.15, color='#5b7553', label='Intercritical (α + γ)')
    ax.fill_between(Mn_range, Ac3s, 950, alpha=0.08, color='#4a7a8c', label='Austenite (γ)')
    ax.fill_between(Mn_range, 400, Ac1s, alpha=0.08, color='#b8860b', label='Ferrite/martensite (α)')
    ax.plot(Mn_range, Ac1s, '-', color='#8b4513', lw=2, label='Ac1')
    ax.plot(Mn_range, Ac3s, '-', color='#2c5f7c', lw=2, label='Ac3')
    if df_lit is not None:
        subset = df_lit[(df_lit['C'] >= pd_c - 0.03) & (df_lit['C'] <= pd_c + 0.03) & (df_lit['f_RA_pct'] > 0)]
        if len(subset) > 0:
            ax.scatter(subset['Mn'], subset['T_celsius'], c='#333', s=15, alpha=0.5, zorder=5, label=f'Literature (C ≈ {pd_c:.2f})')
    ax.set_xlabel('Mn (wt%)'); ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'Fe-xMn-{pd_c:.2f}C pseudo phase diagram')
    ax.set_xlim(4, 12); ax.set_ylim(400, 950); ax.legend(fontsize=8, framealpha=0.8); ax.grid(True, alpha=0.2)
    fig.tight_layout(); st.pyplot(fig); plt.close()
