import numpy as np
import pytest

from blinit.exact import cole_hopf_sine
from blinit.observe import ObsConfig, nested_mask, observe
from blinit.simulator import CallLog, SimConfig, initial_condition, interp_periodic, solve


def test_config_loads_documented_values():
    cfg = SimConfig.from_json()
    assert cfg.nx == 256 and cfg.cfl == 0.25 and cfg.domain == (0.0, 1.0)
    assert ObsConfig.from_json().times == (0.25, 0.5, 0.75, 1.0)


def test_cole_hopf_reduces_to_initial_condition():
    x = np.linspace(0, 1, 50, endpoint=False)
    np.testing.assert_allclose(cole_hopf_sine(x, 0.0, 1.0, 0.05), np.sin(2 * np.pi * x), atol=1e-10)


def test_constant_state_preserved():
    cfg = SimConfig(nx=64)
    out = solve(np.full(64, 0.7), 0.05, [0.3], cfg)
    np.testing.assert_allclose(out, 0.7, atol=1e-13)


def test_mass_conservation_and_finite():
    cfg = SimConfig(nx=128)
    u0 = initial_condition(cfg.xc, [0.9, 0.7, 0.8], [0.3, 2.0, 4.0])
    out = solve(u0, 0.01, [0.25, 1.0], cfg)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out.mean(axis=1), u0.mean(), atol=1e-13)


def test_periodic_shift_equivariance():
    """Shifting the IC by whole cells shifts the solution exactly: checks periodic BCs."""
    cfg = SimConfig(nx=128)
    u0 = initial_condition(cfg.xc, [0.9, 0.4, 0.6], [0.3, 2.0, 4.0])
    a = solve(u0, 0.02, [0.5], cfg)
    b = solve(np.roll(u0, 37), 0.02, [0.5], cfg)
    np.testing.assert_allclose(np.roll(a, 37, axis=1), b, atol=1e-12)


def test_small_amplitude_heat_limit():
    """A -> 0: Burgers -> heat equation with D = nu/pi (checks the viscosity convention)."""
    cfg = SimConfig(nx=256)
    A, nu, t = 1e-6, 0.05, 0.5
    out = solve(A * np.sin(2 * np.pi * cfg.xc), nu, [t], cfg)[0]
    exact = A * np.exp(-(nu / np.pi) * (2 * np.pi) ** 2 * t) * np.sin(2 * np.pi * cfg.xc)
    assert np.max(np.abs(out - exact)) / A < 2e-3
    # the textbook convention (D = nu) would decay far more; make sure we are not using it
    wrong = A * np.exp(-nu * (2 * np.pi) ** 2 * t)
    assert abs(np.max(np.abs(out)) - wrong) / A > 0.1


@pytest.mark.parametrize("nu", [0.02, 0.1])
def test_cole_hopf_grid_convergence(nu):
    errs = []
    for nx in (64, 128, 256):
        cfg = SimConfig(nx=nx)
        out = solve(np.sin(2 * np.pi * cfg.xc), nu, [0.5], cfg)[0]
        errs.append(np.max(np.abs(out - cole_hopf_sine(cfg.xc, 0.5, 1.0, nu))))
    assert errs[0] > errs[1] > errs[2]
    assert errs[1] / errs[2] > 2.5  # at least ~second-order-ish refinement in the resolved regime
    assert errs[2] < 1e-2


def test_call_log_counts():
    log = CallLog()
    observe([0.5, 0.5, 0.5], [0, 0, 0], 0.05, SimConfig(nx=64), ObsConfig(), log)
    observe([0.5, 0.5, 0.5], [0, 0, 0], 0.05, SimConfig(nx=64), ObsConfig(), log)
    assert log.n_calls == 2 and log.n_steps > 0 and log.failures == 0


def test_interp_exact_at_cell_centres_and_periodic():
    cfg = SimConfig(nx=32)
    u = np.sin(2 * np.pi * cfg.xc)
    np.testing.assert_allclose(interp_periodic(u, cfg.xc, cfg), u)
    np.testing.assert_allclose(interp_periodic(u, cfg.xc + 1.0, cfg), u, atol=1e-12)


def test_nested_masks():
    m4, m8, m16 = (nested_mask(k) for k in (4, 8, 16))
    assert m4.sum() == 4 and m8.sum() == 8 and m16.all()
    assert np.all(m8[m4]) and np.all(m16[m8])
