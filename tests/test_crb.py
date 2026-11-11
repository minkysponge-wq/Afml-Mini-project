import numpy as np
import pytest

from blinit.crb import crb_theta, sensitivity
from blinit.observe import nested_mask

AMPS, PHASES, NU = np.array([0.6, 0.3, 0.2]), np.array([0.4, 1.3, 2.9]), 0.03


@pytest.fixture(scope="module")
def sens():
    return sensitivity(AMPS, PHASES, NU, h=1e-3)


def test_derivative_stable_in_step(sens):
    fine = sensitivity(AMPS, PHASES, NU, h=1e-4)
    assert sens.shape == (4, 16)
    assert np.linalg.norm(sens - fine) <= 0.01 * np.linalg.norm(fine)
    # the bound itself moves by less than 1% as well
    m = nested_mask(16)
    assert abs(crb_theta(sens, m, 0.01) / crb_theta(fine, m, 0.01) - 1) < 0.01


def test_bound_scales_with_sigma_squared(sens):
    m = nested_mask(8)
    np.testing.assert_allclose(crb_theta(sens, m, 0.02), 4.0 * crb_theta(sens, m, 0.01), rtol=1e-12)


def test_more_sensors_never_loosen_the_bound(sens):
    b4, b8, b16 = (crb_theta(sens, nested_mask(k), 0.01) for k in (4, 8, 16))
    assert b16 <= b8 <= b4
