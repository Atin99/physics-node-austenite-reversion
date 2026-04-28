"""
Streamlit app for the Physics-Constrained Neural ODE.
Predicts austenite reversion kinetics in medium-Mn steels.

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
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("streamlit not installed. pip install streamlit")
    sys.exit(1)

import torch
from config import get_config
from thermodynamics import get_Ac1_Ac3, get_equilibrium_RA, get_driving_force
from features import compute_Md30, compute_diffusivity, compute_hollomon_jaffe
from model import PhysicsNODE

# --- page config ---
st.set_page_config(
    page_title="PhysicsNODE - Austenite Reversion",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- clean theme with warm neutrals ---
st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Sans Pro', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }
    h1 { color: #2c3e50; font-weight: 600; font-size: 1.8rem; }
    h2 { color: #34495e; font-weight: 600; font-size: 1.3rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
    h3 { color: #4a5568; font-weight: 600; font-size: 1.1rem; }
    .metric-box {
        background: #f7f7f5;
        border: 1px solid #e2e0dc;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-val { font-size: 1.6rem; font-weight: 600; color: #2c3e50; }
    .metric-label { font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; }
    .info-block {
        background: #fafaf8;
        border-left: 3px solid #b8a88a;
        padding: 0.8rem 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #4a5568;
    }
    footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)


# --- model loading ---
@st.cache_resource
def load_model():
    cfg = get_config()
    model = PhysicsNODE(cfg.model)

    # try multiple checkpoint locations
    candidates = [
        REPO_ROOT / "models" / "stage2_fixed_best.pt",
        REPO_ROOT / "models" / "final_best_stage2_fixed.pt",
        REPO_ROOT / "models" / "physics_node_best.pt",
        cfg.checkpoint_dir / "physics_node_best.pt",
    ]
    loaded = False
    for ckpt_path in candidates:
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
            model.load_state_dict(state, strict=False)
            loaded = True
            break

    model.eval()
    return model, cfg, loaded


@st.cache_data
def load_dataset():
    csv_path = REPO_ROOT / "data" / "literature_validation" / "literature_validation.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def predict_curve(model, cfg, comp, T_c, t_max_sec):
    """run the model for one composition + temperature + time schedule."""
    T_K = T_c + 273.15
    f_eq, _ = get_equilibrium_RA(comp, T_c, force_fallback=True)
    dG = get_driving_force(comp, T_c, force_fallback=True)
    D = compute_diffusivity(T_K)
    P = compute_hollomon_jaffe(T_K, max(t_max_sec / 2, 1.0))

    static = torch.tensor([[
        (T_K - cfg.data.T_ref) / cfg.data.T_scale,
        comp['Mn'], comp['C'], comp.get('Al', 0), comp.get('Si', 0),
        np.log10(D + 1e-30), dG / 1000.0, P / 20000.0
    ]], dtype=torch.float32)

    f_eq_t = torch.tensor([[f_eq]], dtype=torch.float32)
    dG_t = torch.tensor([[dG / 1000.0]], dtype=torch.float32)
    t_span = torch.linspace(0, float(t_max_sec), 50)

    mean, lo, hi = model.predict_with_uncertainty(static, f_eq_t, dG_t, t_span, 30)

    Ac1, Ac3 = get_Ac1_Ac3(comp)
    Md = compute_Md30(comp)

    return {
        't_seconds': t_span.numpy(),
        't_hours': t_span.numpy() / 3600,
        'f_RA_mean': mean[0].numpy(),
        'f_RA_lower': lo[0].numpy(),
        'f_RA_upper': hi[0].numpy(),
        'f_eq': f_eq,
        'Md30': Md,
        'D_Mn': D,
        'Ac1': Ac1,
        'Ac3': Ac3,
        'delta_G': dG
    }


def render_metric(label, value):
    st.markdown(f"""<div class="metric-box">
        <div class="metric-val">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================
model, cfg, model_loaded = load_model()
df = load_dataset()

st.title("PhysicsNODE")
st.markdown("Physics-constrained Neural ODE for austenite reversion kinetics in medium-Mn steels")

if not model_loaded:
    st.warning("No trained model checkpoint found. Predictions will be from an untrained model.")

# --- sidebar ---
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    Predicts retained austenite fraction during intercritical annealing
    of medium-Mn steels (3-12 wt% Mn).

    Trained on 125 experimental measurements from 25 published studies.

    **Best metrics:**
    - Val RMSE: 0.157
    - Test RMSE: 0.136
    - Overall R\u00b2: 0.013
    """)
    st.markdown("---")
    st.markdown("### Model")
    st.markdown(f"""
    - Architecture: Latent Neural ODE
    - Solver: Dormand-Prince 4/5
    - Physics: monotonicity, boundary, thermodynamic
    - Parameters: 78,474
    - Checkpoint: {'loaded' if model_loaded else 'not found'}
    """)

# --- tabs ---
tab_predict, tab_sweep, tab_data, tab_phase = st.tabs([
    "Prediction", "Temperature Sweep", "Dataset", "Phase Diagram"
])


# --- tab 1: forward prediction ---
with tab_predict:
    st.header("Forward Prediction")
    st.markdown("Predict RA fraction for a given alloy composition and annealing schedule.")

    col_input, col_output = st.columns([1, 2])

    with col_input:
        st.subheader("Composition (wt%)")
        Mn = st.slider("Mn", 4.0, 12.0, 7.0, 0.1, key="p_mn")
        C = st.slider("C", 0.05, 0.30, 0.10, 0.01, key="p_c")
        Al = st.slider("Al", 0.0, 3.0, 0.0, 0.1, key="p_al")
        Si = st.slider("Si", 0.0, 2.0, 0.0, 0.1, key="p_si")

        st.subheader("Annealing")
        T = st.slider("Temperature (C)", 550, 800, 650, 5, key="p_t")
        t_min = st.slider("Time (minutes)", 1, 300, 60, 1, key="p_time")

        run = st.button("Run prediction", key="p_run")

    with col_output:
        if run:
            comp = {'Mn': Mn, 'C': C, 'Al': Al, 'Si': Si}
            with st.spinner("Solving ODE..."):
                r = predict_curve(model, cfg, comp, T, t_min * 60)

            # metrics row
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric("Final RA", f"{r['f_RA_mean'][-1]:.1%}")
            with c2:
                render_metric("Equilibrium", f"{r['f_eq']:.1%}")
            with c3:
                render_metric("Ac1 / Ac3", f"{r['Ac1']:.0f} / {r['Ac3']:.0f} C")
            with c4:
                render_metric("Md30", f"{r['Md30']:.0f} C")

            # plot
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams.update({'font.size': 10, 'font.family': 'serif'})

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.fill_between(r['t_hours'], r['f_RA_lower'] * 100, r['f_RA_upper'] * 100,
                            alpha=0.2, color='#5b7553', label='95% CI')
            ax.plot(r['t_hours'], r['f_RA_mean'] * 100, '-', color='#5b7553', lw=2, label='Mean prediction')
            ax.axhline(r['f_eq'] * 100, color='#b8860b', ls='--', alpha=0.6,
                       label=f'Equilibrium ({r["f_eq"]:.1%})')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Retained austenite (%)')
            ax.set_title(f'Fe-{Mn:.1f}Mn-{C:.2f}C-{Al:.1f}Al at {T} C', fontsize=11)
            ax.legend(fontsize=9, framealpha=0.8)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

            # thermodynamic details
            with st.expander("Thermodynamic details"):
                st.markdown(f"""
                - Driving force (delta G): {r['delta_G']:.0f} J/mol
                - Mn diffusivity at {T} C: {r['D_Mn']:.2e} m2/s
                - Hollomon-Jaffe parameter: {compute_hollomon_jaffe(T+273.15, t_min*30):.0f}
                """)


# --- tab 2: temperature sweep ---
with tab_sweep:
    st.header("Temperature Sweep")
    st.markdown("Scan across temperatures at fixed time to find optimal annealing conditions.")

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.subheader("Composition (wt%)")
        Mn2 = st.slider("Mn", 4.0, 12.0, 7.0, 0.1, key="s_mn")
        C2 = st.slider("C", 0.05, 0.30, 0.10, 0.01, key="s_c")
        Al2 = st.slider("Al", 0.0, 3.0, 0.0, 0.1, key="s_al")
        Si2 = st.slider("Si", 0.0, 2.0, 0.0, 0.1, key="s_si")
        t_hold = st.slider("Hold time (minutes)", 10, 300, 60, 10, key="s_time")
        run2 = st.button("Run sweep", key="s_run")

    with col_out:
        if run2:
            comp2 = {'Mn': Mn2, 'C': C2, 'Al': Al2, 'Si': Si2}
            Ac1, Ac3 = get_Ac1_Ac3(comp2)
            T_range = np.linspace(max(Ac1 - 20, 550), min(Ac3 + 20, 800), 20)

            results = []
            progress = st.progress(0)
            for i, Ti in enumerate(T_range):
                try:
                    ri = predict_curve(model, cfg, comp2, float(Ti), t_hold * 60)
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
                temps = [r['T'] for r in results]
                ras = [r['RA'] for r in results]
                feqs = [r['f_eq'] for r in results]

                ax.plot(temps, ras, 'o-', color='#5b7553', lw=2, markersize=5, label='Predicted RA')
                ax.plot(temps, feqs, '--', color='#b8860b', alpha=0.6, label='Equilibrium')
                ax.axvline(Ac1, color='#cc4444', ls=':', alpha=0.5, label=f'Ac1 = {Ac1:.0f} C')
                ax.axvline(Ac3, color='#4444cc', ls=':', alpha=0.5, label=f'Ac3 = {Ac3:.0f} C')
                ax.set_xlabel('Temperature (C)')
                ax.set_ylabel('Retained austenite (%)')
                ax.set_title(f'Fe-{Mn2:.1f}Mn-{C2:.2f}C at {t_hold} min hold', fontsize=11)
                ax.legend(fontsize=9, framealpha=0.8)
                ax.set_ylim(bottom=0)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

                best = max(results, key=lambda x: x['RA'])
                st.markdown(f'<div class="info-block">Peak RA: <b>{best["RA"]:.1f}%</b> at <b>{best["T"]:.0f} C</b></div>',
                            unsafe_allow_html=True)


# --- tab 3: dataset explorer ---
with tab_data:
    st.header("Literature Dataset")

    if df is not None:
        st.markdown(f"125 experimental measurements from 25 published studies (2010-2024).")

        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric("Data Points", str(len(df)))
        with col2:
            render_metric("Studies", str(df['study_id'].nunique()))
        with col3:
            render_metric("Alloys", str(len(df.groupby(['Mn', 'C', 'Al']))))

        # filters
        st.subheader("Filter")
        fc1, fc2 = st.columns(2)
        with fc1:
            sel_study = st.multiselect("Study", sorted(df['study_id'].unique()), key="d_study")
        with fc2:
            sel_method = st.multiselect("Method", sorted(df['method'].unique()), key="d_method")

        filtered = df.copy()
        if sel_study:
            filtered = filtered[filtered['study_id'].isin(sel_study)]
        if sel_method:
            filtered = filtered[filtered['method'].isin(sel_method)]

        st.dataframe(
            filtered[['study_id', 'Mn', 'C', 'Al', 'T_celsius', 't_seconds', 'f_RA_pct', 'method', 'data_quality', 'doi']],
            height=400, use_container_width=True
        )

        # scatter plot
        st.subheader("Data Distribution")
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams.update({'font.size': 10, 'font.family': 'serif'})

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # T vs RA colored by Mn
        sc = axes[0].scatter(filtered['T_celsius'], filtered['f_RA_pct'],
                             c=filtered['Mn'], cmap='YlOrBr', s=30, alpha=0.7, edgecolors='#555', linewidth=0.3)
        axes[0].set_xlabel('Temperature (C)')
        axes[0].set_ylabel('RA (%)')
        axes[0].set_title('RA vs Temperature')
        plt.colorbar(sc, ax=axes[0], label='Mn (wt%)')

        # Mn vs RA colored by T
        sc2 = axes[1].scatter(filtered['Mn'], filtered['f_RA_pct'],
                              c=filtered['T_celsius'], cmap='YlOrBr', s=30, alpha=0.7, edgecolors='#555', linewidth=0.3)
        axes[1].set_xlabel('Mn (wt%)')
        axes[1].set_ylabel('RA (%)')
        axes[1].set_title('RA vs Mn Content')
        plt.colorbar(sc2, ax=axes[1], label='T (C)')

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.error("Dataset CSV not found.")


# --- tab 4: phase diagram ---
with tab_phase:
    st.header("Pseudo Phase Diagram")
    st.markdown("Ac1-Ac3 boundaries as function of Mn content, computed from empirical correlations.")

    pd_c = st.slider("Carbon content (wt%)", 0.05, 0.30, 0.10, 0.01, key="pd_c")

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({'font.size': 10, 'font.family': 'serif'})

    fig, ax = plt.subplots(figsize=(8, 5))
    Mn_range = np.linspace(4, 12, 60)
    Ac1s, Ac3s = zip(*[get_Ac1_Ac3({'Mn': m, 'C': pd_c}) for m in Mn_range])
    Ac1s, Ac3s = np.array(Ac1s), np.array(Ac3s)

    ax.fill_between(Mn_range, Ac1s, Ac3s, alpha=0.15, color='#5b7553', label='Intercritical (alpha + gamma)')
    ax.fill_between(Mn_range, Ac3s, 950, alpha=0.08, color='#4a7a8c', label='Austenite (gamma)')
    ax.fill_between(Mn_range, 400, Ac1s, alpha=0.08, color='#b8860b', label='Ferrite/martensite (alpha)')
    ax.plot(Mn_range, Ac1s, '-', color='#8b4513', lw=2, label='Ac1')
    ax.plot(Mn_range, Ac3s, '-', color='#2c5f7c', lw=2, label='Ac3')

    # overlay literature data points if available
    if df is not None:
        subset = df[(df['C'] >= pd_c - 0.03) & (df['C'] <= pd_c + 0.03) & (df['f_RA_pct'] > 0)]
        if len(subset) > 0:
            ax.scatter(subset['Mn'], subset['T_celsius'], c='#333', s=15, alpha=0.5,
                       zorder=5, label=f'Literature data (C ~ {pd_c:.2f})')

    ax.set_xlabel('Mn (wt%)')
    ax.set_ylabel('Temperature (C)')
    ax.set_title(f'Fe-xMn-{pd_c:.2f}C pseudo phase diagram')
    ax.set_xlim(4, 12)
    ax.set_ylim(400, 950)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
