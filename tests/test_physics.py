import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config, PhysicalConstants
from features import compute_diffusivity, compute_Md30, compute_JMAK, compute_JMAK_rate, compute_hollomon_jaffe, compute_k_arrhenius, featurize_sample
from thermodynamics import get_Ac1_Ac3, get_equilibrium_RA, get_driving_force, validate_ICA_temperature
from model import PhysicsNODE, physics_gate


class TestDiffusivity:
    def test_arrhenius_increases_with_T(self):
        assert compute_diffusivity(1023.15) > compute_diffusivity(873.15)

    def test_order_of_magnitude(self):
        D = compute_diffusivity(923.15)
        assert 1e-20 < D < 1e-15

    def test_positive(self):
        for T in [573.15, 773.15, 1073.15, 1273.15, 1773.15]:
            assert compute_diffusivity(T) > 0

    def test_C_faster_than_Mn(self):
        T = 923.15
        assert compute_diffusivity(T, "C_austenite") > compute_diffusivity(T, "Mn_austenite") * 1e4

    def test_vectorized(self):
        T = np.array([873.15, 923.15, 973.15, 1023.15])
        D = compute_diffusivity(T)
        assert D.shape == (4,) and np.all(np.diff(D) > 0)


class TestMd30:
    def test_decreases_with_Mn(self):
        assert compute_Md30({'Mn': 12, 'C': 0.1}) < compute_Md30({'Mn': 4, 'C': 0.1})

    def test_decreases_with_C(self):
        assert compute_Md30({'Mn': 7, 'C': 0.3}) < compute_Md30({'Mn': 7, 'C': 0.05})

    def test_reasonable_range(self):
        M = compute_Md30({'Mn': 7, 'C': 0.1, 'Si': 0.5})
        assert -50 < M < 500


class TestJMAK:
    def test_monotonic(self):
        t = np.linspace(0, 10000, 200)
        f = compute_JMAK(t, 1e-6, 2.0, 0.35)
        assert np.all(np.diff(f) >= -1e-10)

    def test_bounded(self):
        t = np.linspace(0, 100000, 1000)
        f = compute_JMAK(t, 1e-4, 2.0, 0.40)
        assert f.max() <= 0.40 + 1e-10

    def test_starts_at_zero(self):
        f = compute_JMAK(np.array([0.0, 100.0, 1000.0]), 1e-6, 2.0, 0.35)
        assert abs(f[0]) < 1e-10

    def test_approaches_f_eq(self):
        f = compute_JMAK(np.array([0, 1e6]), 1e-4, 2.0, 0.35)
        assert abs(f[-1] - 0.35) < 0.001

    def test_rate_non_negative(self):
        t = np.linspace(1, 10000, 200)
        assert np.all(compute_JMAK_rate(t, 1e-6, 2.0, 0.35) >= -1e-15)


class TestThermodynamics:
    def test_Ac1_below_Ac3(self):
        for Mn in [4, 6, 8, 10, 12]:
            Ac1, Ac3 = get_Ac1_Ac3({'Mn': Mn, 'C': 0.1})
            assert Ac1 < Ac3

    def test_Mn_depresses_Ac1(self):
        assert get_Ac1_Ac3({'Mn': 12, 'C': 0.1})[0] < get_Ac1_Ac3({'Mn': 4, 'C': 0.1})[0]

    def test_f_eq_in_range(self):
        for T in [550, 600, 650, 700, 750, 800]:
            f, _ = get_equilibrium_RA({'Mn': 7, 'C': 0.1}, T)
            assert 0 <= f <= 1

    def test_f_eq_increases_with_T(self):
        comp = {'Mn': 7, 'C': 0.1}
        Ac1, Ac3 = get_Ac1_Ac3(comp)
        if Ac1 + 20 < Ac3 - 20:
            f_lo, _ = get_equilibrium_RA(comp, Ac1 + 20)
            f_hi, _ = get_equilibrium_RA(comp, Ac3 - 20)
            assert f_hi >= f_lo

    def test_driving_force_reasonable(self):
        comp = {'Mn': 7, 'C': 0.1}
        Ac1, Ac3 = get_Ac1_Ac3(comp)
        assert abs(get_driving_force(comp, (Ac1 + Ac3) / 2)) < 50000

    def test_ICA_validation(self):
        comp = {'Mn': 7, 'C': 0.1}
        Ac1, Ac3 = get_Ac1_Ac3(comp)
        assert not validate_ICA_temperature(comp, Ac1 - 50)['valid']
        assert not validate_ICA_temperature(comp, Ac3 + 50)['valid']
        assert validate_ICA_temperature(comp, (Ac1 + Ac3) / 2)['valid']


class TestPhysicsGate:
    def test_nonzero_at_f_zero(self):
        assert physics_gate(torch.tensor([[0.0]]), torch.tensor([[0.35]]), torch.tensor([[-0.5]]), 0.005).item() > 0

    def test_zero_at_equilibrium(self):
        assert physics_gate(torch.tensor([[0.35]]), torch.tensor([[0.35]]), torch.tensor([[-0.5]])).item() < 1e-6

    def test_decreases_near_eq(self):
        f_eq, dG = torch.tensor([[0.35]]), torch.tensor([[-0.5]])
        g = [physics_gate(torch.tensor([[v]]), f_eq, dG).item() for v in [0.0, 0.1, 0.2, 0.3, 0.34]]
        assert g[-1] < g[1]

    def test_non_negative(self):
        for v in [0.0, 0.1, 0.2, 0.35, 0.5]:
            assert physics_gate(torch.tensor([[v]]), torch.tensor([[0.35]]), torch.tensor([[-0.5]])).item() >= 0


class TestHollomonJaffe:
    def test_increases_with_T(self):
        assert compute_hollomon_jaffe(1023.15, 3600) > compute_hollomon_jaffe(873.15, 3600)

    def test_increases_with_time(self):
        assert compute_hollomon_jaffe(923.15, 7200) > compute_hollomon_jaffe(923.15, 60)


class TestFeatures:
    def test_shape(self):
        f = featurize_sample({'Mn': 7, 'C': 0.1, 'Al': 1.5, 'Si': 0.5}, 650, 1800, 0.15, 0.35, -500)
        assert f.shape == (10,)

    def test_finite(self):
        f = featurize_sample({'Mn': 7, 'C': 0.1, 'Al': 1.5, 'Si': 0.5}, 650, 1800, 0.15, 0.35, -500)
        assert np.all(np.isfinite(f))

    def test_at_t_zero(self):
        f = featurize_sample({'Mn': 7, 'C': 0.1, 'Al': 1.5, 'Si': 0.5}, 650, 0.0, 0.0, 0.35, -500)
        assert np.all(np.isfinite(f))


class TestModel:
    def test_creates(self):
        assert PhysicsNODE(get_config().model) is not None

    def test_param_count(self):
        n = sum(p.numel() for p in PhysicsNODE(get_config().model).parameters())
        assert 5000 < n < 500000

    def test_forward(self):
        cfg = get_config()
        cfg.model.max_num_steps = 5000
        model = PhysicsNODE(cfg.model)
        model.eval()
        static = torch.tensor([[2.5, 7.0, 0.1, 1.5, 0.5, -17.5, -0.5, 0.8], [2.5, 7.0, 0.1, 1.5, 0.5, -17.5, -0.5, 0.8]])
        f_eq = torch.full((2, 1), 0.35)
        dG = torch.full((2, 1), -0.5)
        t_span = torch.linspace(0, 100, 10)
        with torch.no_grad():
            out = model(static, f_eq, dG, t_span)
        assert out.shape == (2, 10) and torch.all(torch.isfinite(out))

    def test_forward_with_sample_specific_time_grids(self):
        cfg = get_config()
        cfg.model.max_num_steps = 5000
        model = PhysicsNODE(cfg.model)
        model.eval()
        static = torch.tensor([[2.5, 7.0, 0.1, 1.5, 0.5, -17.5, 5.0, 0.8], [2.8, 8.5, 0.12, 1.0, 0.3, -17.0, 3.0, 0.9]])
        f_eq = torch.tensor([[0.35], [0.42]])
        dG = torch.tensor([[5.0], [3.0]])
        t_span = torch.tensor([
            [0.0, 10.0, 100.0, 500.0, 3600.0],
            [0.0, 5.0, 50.0, 250.0, 1800.0],
        ])
        with torch.no_grad():
            out = model(static, f_eq, dG, t_span)
        assert out.shape == (2, 5) and torch.all(torch.isfinite(out))

    def test_initialized_kinetics_are_monotone_and_nonflat(self):
        cfg = get_config()
        cfg.model.max_num_steps = 5000
        model = PhysicsNODE(cfg.model)
        model.eval()
        static = torch.tensor([[2.5, 7.0, 0.1, 1.5, 0.5, -17.5, 5.0, 0.8]])
        f_eq = torch.tensor([[0.35]])
        dG = torch.tensor([[5.0]])
        t_span = torch.linspace(0, 3600, 20)
        with torch.no_grad():
            out = model(static, f_eq, dG, t_span)[0]
        assert torch.all(out[1:] >= out[:-1] - 1e-6)
        assert out[-1] > out[0] + 0.01
        assert out[1] < out[-1] * 0.95

    def test_composition_encoder(self):
        from model import CompositionEncoder
        enc = CompositionEncoder(5, 32, 4, False)
        x = torch.randn(4, 5)
        out = enc(x)
        assert out.shape == (4, 32)



    def test_swag_collect(self):
        from model import SWAG
        m = PhysicsNODE(get_config().model)
        swag = SWAG(m, rank=3)
        swag.collect()
        assert swag.n_collected == 1


class TestKArrhenius:
    def test_increases_with_T(self):
        assert compute_k_arrhenius(1023.15, 7, 0.1) > compute_k_arrhenius(873.15, 7, 0.1)

    def test_positive(self):
        assert compute_k_arrhenius(923.15, 7, 0.1) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
