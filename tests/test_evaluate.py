import numpy as np
import pytest

from blinit.experiments.evaluate import METHOD_GROUPS, parse_methods, run_variant
from blinit.observe import ObsConfig, nested_mask, observe
from blinit.simulator import SimConfig

CFG = dict(brent=dict(xatol=1e-2, maxiter=100),
           lsq_refine=dict(xtol=1e-2, ftol=1e-10, diff_step=1e-6, max_nfev=100), grid_k=5)


def _case():
    amps, phases, nu = np.array([0.6, 0.3, 0.2]), np.array([0.4, 1.3, 2.9]), 0.03
    y = observe(amps, phases, nu, SimConfig.from_json(), ObsConfig.from_json())
    return amps, phases, y, nested_mask(16), nu


def test_parse_methods():
    assert parse_methods(None) == METHOD_GROUPS
    assert parse_methods("mlp, brent") == ("brent", "mlp")
    with pytest.raises(ValueError):
        parse_methods("brent,newton")


def test_run_variant_selected_methods_only():
    amps, phases, y, mask, nu = _case()
    rows = run_variant((0, amps, phases, y, mask, nu, {}, CFG, ("brent",)))
    assert [r["method"] for r in rows] == ["brent"]
    assert rows[0]["rel_err"] < 0.05 and rows[0]["variant"] == 0
    rows = run_variant((0, amps, phases, y, mask, nu, {"mlp_s0": float(np.log(0.04))}, CFG, ("midpoint", "mlp")))
    assert [r["method"] for r in rows] == ["midpoint+lsq", "mlp_s0+lsq"]
