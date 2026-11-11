"""Cramer-Rao bound for theta = log(nu) from sparse, noisy sensor readings.

Model: y = f(theta) + eps, eps ~ N(0, sigma^2) independently on every available reading.
Fisher information  I(theta) = (1 / sigma^2) * sum_{available i} (d f_i / d theta)^2,
and any unbiased estimator has Var(theta_hat) >= 1 / I(theta).
Because theta = log(nu), a standard deviation of 0.01 in theta is about a 1% relative error in nu.
"""
from __future__ import annotations

import numpy as np

from .observe import ObsConfig
from .simulator import SimConfig, initial_condition, interp_periodic, solve


def sensitivity(amps, phases, nu, h=1e-3, sim: SimConfig | None = None, obs: ObsConfig | None = None,
                modes=(1, 2, 3)):
    """d f / d theta at all sensor slots and times, shape [n_times, n_slots], by central difference in theta."""
    sim = sim or SimConfig.from_json()
    obs = obs or ObsConfig.from_json()
    u0 = initial_condition(sim.xc, amps, phases, modes=modes, length=sim.length)
    th = np.log(nu)
    f = [interp_periodic(solve(u0, float(np.exp(th + s * h)), obs.times, sim), obs.sensor_x, sim) for s in (1, -1)]
    return (f[0] - f[1]) / (2.0 * h)


def fisher_theta(sens, mask, sigma):
    """Fisher information for theta from the readings selected by mask (same sensors at all times)."""
    return float(np.sum(sens[:, np.asarray(mask, bool)] ** 2)) / float(sigma) ** 2


def crb_theta(sens, mask, sigma):
    """Lower bound on Var(theta_hat) for one sensor mask: 1 / I(theta)."""
    return 1.0 / fisher_theta(sens, mask, sigma)
