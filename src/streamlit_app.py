import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

if STREAMLIT_AVAILABLE:
    import torch
    from config import get_config
    from thermodynamics import get_Ac1_Ac3, get_equilibrium_RA, get_driving_force
    from features import compute_Md30, compute_diffusivity, compute_hollomon_jaffe
    from model import PhysicsNODE

    st.set_page_config(page_title="PhysicsNODE", page_icon="🔬", layout="wide")

    st.markdown("""<style>
        .main-header{font-size:2.2rem;font-weight:bold;color:#1E3A5F;margin-bottom:.5rem}
        .sub-header{font-size:1.1rem;color:#5A6C7D;margin-bottom:2rem}
        .stTabs [data-baseweb="tab"]{font-size:1.1rem;font-weight:600}
    </style>""", unsafe_allow_html=True)

    @st.cache_resource
    def load_model():
        cfg = get_config()
        model = PhysicsNODE(cfg.model)
        p = cfg.checkpoint_dir / "physics_node_best.pt"
        if p.exists():
            ckpt = torch.load(p, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model'])
            st.sidebar.success("Trained model loaded")
        else:
            st.sidebar.warning("Using untrained model")
        model.eval()
        return model, cfg

    def predict(model, cfg, comp, T_c, t_max):
        T_K = T_c + 273.15
        f_eq, _ = get_equilibrium_RA(comp, T_c, force_fallback=True)
        dG = get_driving_force(comp, T_c, force_fallback=True)
        D = compute_diffusivity(T_K)
        P = compute_hollomon_jaffe(T_K, max(t_max/2, 1.0))
        Ac1, Ac3 = get_Ac1_Ac3(comp)
        Md = compute_Md30(comp)

        static = torch.tensor([[(T_K - cfg.data.T_ref) / cfg.data.T_scale, comp['Mn'], comp['C'],
                                 comp.get('Al', 0), comp.get('Si', 0), np.log10(D + 1e-30),
                                 dG / 1000.0, P / 20000.0]], dtype=torch.float32)
        f_eq_t = torch.tensor([[f_eq]], dtype=torch.float32)
        dG_t = torch.tensor([[dG / 1000.0]], dtype=torch.float32)
        t_span = torch.linspace(0, float(t_max), 50)
        mean, lo, hi = model.predict_with_uncertainty(static, f_eq_t, dG_t, t_span, 50)
        return {'t_hours': t_span.numpy()/3600, 'f_RA_mean': mean[0].numpy(), 'f_RA_lower': lo[0].numpy(),
                'f_RA_upper': hi[0].numpy(), 'f_eq': f_eq, 'Md30': Md, 'D_Mn': D, 'Ac1': Ac1, 'Ac3': Ac3, 'delta_G': dG}

    model, cfg = load_model()
    st.markdown('<div class="main-header">🔬 PhysicsNODE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Attention-Conditioned Latent Neural ODE for Austenite Reversion in Medium-Mn Steels</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔮 Forward Prediction", "🎯 Inverse Design", "📊 Phase Diagram"])

    with tab1:
        st.header("Forward Prediction")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Composition (wt%)")
            Mn = st.slider("Mn", 4.0, 12.0, 7.0, 0.1, key="fMn")
            C = st.slider("C", 0.05, 0.30, 0.10, 0.01, key="fC")
            Al = st.slider("Al", 0.0, 3.0, 1.5, 0.1, key="fAl")
            Si = st.slider("Si", 0.0, 2.0, 0.5, 0.1, key="fSi")
            st.subheader("Schedule")
            T = st.slider("T (°C)", 550, 800, 650, 5, key="fT")
            tm = st.slider("t (min)", 1, 180, 60, 1, key="ft")
            btn = st.button("Predict", type="primary", key="fbtn")
        with c2:
            if btn:
                comp = {'Mn': Mn, 'C': C, 'Al': Al, 'Si': Si}
                with st.spinner("Running..."):
                    r = predict(model, cfg, comp, T, tm*60)
                fRA = r['f_RA_mean'][-1]
                m1, m2, m3 = st.columns(3)
                m1.metric("RA", f"{fRA:.1%}")
                m2.metric("f_eq", f"{r['f_eq']:.1%}")
                m3.metric("Md30", f"{r['Md30']:.0f}°C")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.fill_between(r['t_hours'], r['f_RA_lower'], r['f_RA_upper'], alpha=0.3, color='#667eea')
                ax.plot(r['t_hours'], r['f_RA_mean'], '-', color='#667eea', lw=2)
                ax.axhline(r['f_eq'], color='red', ls='--', alpha=0.5, label=f'f_eq={r["f_eq"]:.2f}')
                ax.set_xlabel('Time (h)')
                ax.set_ylabel('RA fraction')
                ax.set_title(f'Fe-{Mn:.1f}Mn-{C:.2f}C at {T}°C')
                ax.legend(fontsize=8)
                st.pyplot(fig)

    with tab2:
        st.header("Inverse Design")
        c1, c2 = st.columns([1, 2])
        with c1:
            Mn2 = st.slider("Mn", 4.0, 12.0, 7.0, 0.1, key="iMn")
            C2 = st.slider("C", 0.05, 0.30, 0.10, 0.01, key="iC")
            Al2 = st.slider("Al", 0.0, 3.0, 1.5, 0.1, key="iAl")
            Si2 = st.slider("Si", 0.0, 2.0, 0.5, 0.1, key="iSi")
            target = st.slider("Target RA (%)", 5, 50, 30, 1, key="itgt")
            btn2 = st.button("Optimize", type="primary", key="ibtn")
        with c2:
            if btn2:
                comp2 = {'Mn': Mn2, 'C': C2, 'Al': Al2, 'Si': Si2}
                Ac1, Ac3 = get_Ac1_Ac3(comp2)
                with st.spinner("Searching..."):
                    T_r = np.linspace(max(Ac1+10, 575), min(Ac3-10, 750), 12)
                    t_r = np.logspace(np.log10(60), np.log10(7200), 12)
                    grid = []
                    for Ti in T_r:
                        for ti in t_r:
                            try:
                                ri = predict(model, cfg, comp2, Ti, ti)
                                f = ri['f_RA_mean'][-1]
                                grid.append({'T': Ti, 't': ti, 'f_RA': f, 'err': abs(f - target/100)})
                            except Exception:
                                continue
                if grid:
                    grid.sort(key=lambda x: x['err'])
                    st.subheader("Top 5 Schedules")
                    for i, g in enumerate(grid[:5]):
                        st.write(f"**#{i+1}**: T={g['T']:.0f}°C, t={g['t']/60:.1f}min → RA={g['f_RA']:.1%}")

    with tab3:
        st.header("Phase Diagram")
        C3 = st.slider("C (wt%)", 0.05, 0.30, 0.10, 0.01, key="pdC")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        Mn_r = np.linspace(4, 12, 50)
        Ac1s, Ac3s = zip(*[get_Ac1_Ac3({'Mn': m, 'C': C3}) for m in Mn_r])
        Ac1s, Ac3s = np.array(Ac1s), np.array(Ac3s)
        ax.fill_between(Mn_r, Ac1s, Ac3s, alpha=0.3, color='#667eea', label='α+γ')
        ax.fill_between(Mn_r, Ac3s, 950, alpha=0.15, color='#56B4E9', label='γ')
        ax.fill_between(Mn_r, 400, Ac1s, alpha=0.15, color='#E69F00', label='α')
        ax.plot(Mn_r, Ac1s, '-', color='#D55E00', lw=2, label='Ac1')
        ax.plot(Mn_r, Ac3s, '-', color='#0072B2', lw=2, label='Ac3')
        ax.set_xlabel('Mn (wt%)')
        ax.set_ylabel('T (°C)')
        ax.set_title(f'Fe-xMn-{C3:.2f}C')
        ax.set_xlim(4, 12)
        ax.set_ylim(400, 950)
        ax.legend(fontsize=9)
        st.pyplot(fig)

    with st.sidebar:
        st.markdown("---")
        st.markdown("**PhysicsNODE v2.0**\nAttention-conditioned Latent ODE\nSWAG + Concrete Dropout UQ")
