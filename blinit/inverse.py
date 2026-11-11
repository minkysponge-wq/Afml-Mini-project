"""Inverse problem and classical estimators.

All optimisation is over theta = log(nu) within [log 0.01, log 0.1]. Every method uses
the same Problem object, so forward solves (including finite-difference Jacobian solves)
are counted identically. Ground-truth nu is never visible to a Problem.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from .observe import ObsConfig
from .simulator import CallLog, SimConfig, initial_condition, interp_periodic, solve

NU_LO, NU_HI = 0.01, 0.1
TH_LO, TH_HI = np.log(NU_LO), np.log(NU_HI)
FAIL_RESIDUAL = 1e3


@dataclass
class Problem:
    amps: np.ndarray
    phases: np.ndarray
    y_obs: np.ndarray            # [n_times, n_slots]; entries outside mask are ignored
    mask: np.ndarray             # [n_slots] bool (same sensors at all times in the pilot)
    sim: SimConfig = field(default_factory=SimConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    log: CallLog = field(default_factory=CallLog)
    n_requests: int = 0
    _cache: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)  # (theta, objective, cumulative solver steps) per actual solve
    modes: tuple = (1, 2, 3)     # sine modes of the (known) initial condition

    def __post_init__(self):
        self._u0 = initial_condition(self.sim.xc, self.amps, self.phases, modes=self.modes, length=self.sim.length)
        self._xs = self.obs.sensor_x[self.mask]

    def residual(self, theta):
        th = float(np.atleast_1d(theta)[0])
        self.n_requests += 1
        if th in self._cache:   # exact repeats only; same policy for every method
            return self._cache[th]
        try:
            traj = solve(self._u0, np.exp(th), self.obs.times, self.sim, self.log)
            r = (interp_periodic(traj, self._xs, self.sim) - self.y_obs[:, self.mask]).ravel()
        except FloatingPointError:
            r = np.full(len(self.obs.times) * self.mask.sum(), FAIL_RESIDUAL)
        self._cache[th] = r
        self.trace.append((th, 0.5 * float(r @ r), self.log.n_steps))
        return r

    def objective(self, theta):
        r = self.residual(theta)
        return 0.5 * float(r @ r)


def _clip_theta(th):
    return float(np.clip(th, TH_LO + 1e-9, TH_HI - 1e-9))


def refine_lsq(p: Problem, theta0, xtol=1e-4, ftol=1e-10, diff_step=1e-6, max_nfev=100):
    """Shared bounded least-squares refinement (TRF, 2-point finite-difference Jacobian)."""
    res = least_squares(p.residual, [_clip_theta(theta0)], bounds=([TH_LO], [TH_HI]),
                        method="trf", jac="2-point", diff_step=diff_step,
                        xtol=xtol, ftol=ftol, gtol=1e-12, max_nfev=max_nfev, x_scale=[1.0])
    return float(res.x[0]), dict(status=int(res.status), success=bool(res.success),
                                 nfev=int(res.nfev), njev=int(res.njev or 0))


def coarse_grid_theta(p: Problem, k=5):
    grid = np.linspace(TH_LO, TH_HI, k)
    vals = [p.objective(t) for t in grid]
    return float(grid[int(np.argmin(vals))])


def brent_bounded(p: Problem, xatol=1e-4, maxiter=100):
    res = minimize_scalar(p.objective, bounds=(TH_LO, TH_HI), method="bounded",
                          options=dict(xatol=xatol, maxiter=maxiter))
    return float(res.x), dict(status=int(res.status), success=bool(res.success), nfev=int(res.nfev))


def brent_bracket(p, center, width, xatol=1e-3, maxiter=100, edge_tol=None):
    """Brent restricted to [center - width, center + width] (clipped to the global bounds).

    If the result lies within edge_tol (default 2 * xatol) of an interior bracket edge, the minimum may be
    outside the bracket, so Brent is re-run on the full bounds. Every solve, including the fallback, is
    counted by the shared Problem (exact repeats are memoised, as for every method).
    """
    lo, hi = max(TH_LO, center - width), min(TH_HI, center + width)
    edge_tol = 2 * xatol if edge_tol is None else edge_tol
    res = minimize_scalar(p.objective, bounds=(lo, hi), method="bounded",
                          options=dict(xatol=xatol, maxiter=maxiter))
    th = float(res.x)
    at_edge = (lo > TH_LO and th - lo < edge_tol) or (hi < TH_HI and hi - th < edge_tol)
    info = dict(bracket_lo=lo, bracket_hi=hi, fallback=bool(at_edge), status=int(res.status),
                success=bool(res.success))
    if at_edge:
        th, finfo = brent_bounded(p, xatol=xatol, maxiter=maxiter)
        info.update(status=finfo["status"], success=finfo["success"])
    return th, info


def run_method(name, make_problem, init_theta=None, refine_kw=None, brent_kw=None, grid_k=5,
               bracket_width=None):
    """Run one method on a fresh Problem; return estimate and cost accounting."""
    p = make_problem()
    t0 = time.perf_counter()
    info = {}
    if name == "brent":
        th, info = brent_bounded(p, **(brent_kw or {}))
    elif name.endswith("+brent_bracket"):    # learned bracket for direct search (follow-up method)
        th, info = brent_bracket(p, float(init_theta), bracket_width, **(brent_kw or {}))
        info.update(theta0=float(init_theta), bracket_width=bracket_width)
    else:
        if name == "midpoint+lsq":
            th0 = 0.5 * (TH_LO + TH_HI)
        elif name == f"grid{grid_k}+lsq":
            th0 = coarse_grid_theta(p, grid_k)
        elif init_theta is not None:          # externally supplied init (oracle / learned / kNN)
            th0 = float(init_theta)
        else:
            raise ValueError(name)
        info["n_solves_init"] = p.log.n_calls
        info["theta0"] = th0
        th, rinfo = refine_lsq(p, th0, **(refine_kw or {}))
        info.update(rinfo)
    wall = time.perf_counter() - t0
    out = dict(method=name, theta_hat=th, nu_hat=float(np.exp(th)),
               n_solves=p.log.n_calls, n_requests=p.n_requests, solver_steps=p.log.n_steps,
               solver_seconds=p.log.seconds, wall_seconds=wall, sim_failures=p.log.failures,
               trace=json.dumps([[round(a, 12), b, c] for a, b, c in p.trace]), **info)
    out["objective"] = p.objective(th)  # post-hoc; normally a cache hit, excluded from cost
    return out
