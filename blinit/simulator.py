"""1D viscous Burgers forward solver.

Independent NumPy re-implementation of the finite-volume scheme used by the
PDEBench Burgers generator (Takamoto et al., 2022): MUSCL reconstruction with
the PDEBench van-Leer-type limiter, Rusanov/Murman advective flux, central
diffusive flux and a two-stage midpoint time integrator. Equation convention
(as in PDEBench):

    u_t + (u^2 / 2)_x = (nu / pi) u_xx,   x in [0, 1), periodic.

No PDEBench source code is copied (its licence restricts derivatives); the
algorithm is reproduced from its published description and source reading.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@dataclass(frozen=True)
class SimConfig:
    nx: int = 256
    cfl: float = 0.25
    second_order: bool = True
    domain: tuple = (0.0, 1.0)

    @classmethod
    def from_json(cls, path=CONFIG_DIR / "simulator_v1.json", **overrides):
        c = json.loads(Path(path).read_text())
        kw = dict(nx=c["nx"], cfl=c["cfl"], second_order=c["second_order"],
                  domain=tuple(c["domain"]))
        kw.update(overrides)
        return cls(**kw)

    @property
    def length(self):
        return self.domain[1] - self.domain[0]

    @property
    def dx(self):
        return self.length / self.nx

    @property
    def xc(self):
        return self.domain[0] + (np.arange(self.nx) + 0.5) * self.dx


@dataclass
class CallLog:
    """Counts every forward solve; shared by all inverse methods."""
    n_calls: int = 0
    n_steps: int = 0
    seconds: float = 0.0
    failures: int = 0
    records: list = field(default_factory=list)
    keep_records: bool = False

    def add(self, nu, steps, secs, ok):
        self.n_calls += 1
        self.n_steps += steps
        self.seconds += secs
        self.failures += (not ok)
        if self.keep_records:
            self.records.append((float(nu), int(steps), float(secs), bool(ok)))


def initial_condition(x, amps, phases, modes=(1, 2, 3), length=1.0):
    amps, phases = np.asarray(amps, float), np.asarray(phases, float)
    k = 2.0 * np.pi * np.asarray(modes, float)[:, None] / length
    return np.sum(amps[:, None] * np.sin(k * x[None, :] + phases[:, None]), axis=0)


def _limiter(a, b, c, alpha=2.0):
    # PDEBench VLlimiter: zero at extrema, else min(alpha*min(|a|,|b|), |c|) with sign(c)
    return np.sign(c) * (0.5 + 0.5 * np.sign(a * b)) * np.minimum(
        alpha * np.minimum(np.abs(a), np.abs(b)), np.abs(c))


def _flux(u, D, dx_inv, second_order):
    """Numerical flux at the N+1 interfaces; f[k] is the left face of cell k."""
    ug = np.concatenate([u[-2:], u, u[:2]])  # two ghost cells per side
    n = u.size
    if second_order:
        du_l = ug[1:n + 3] - ug[0:n + 2]
        du_r = ug[2:n + 4] - ug[1:n + 3]
        du_m = 0.5 * (ug[2:n + 4] - ug[0:n + 2])
        g = _limiter(du_l, du_r, du_m)
    else:
        g = 0.0
    mid = ug[1:n + 3]
    face_l = mid - 0.5 * g  # left face value of cells 1..n+2 (ghost-indexed)
    face_r = mid + 0.5 * g  # right face value
    ul = face_r[0:n + 1]    # state left of interface (right face of cell k-1)
    ur = face_l[1:n + 2]    # state right of interface (left face of cell k)
    f = 0.5 * (0.5 * ul ** 2 + 0.5 * ur ** 2 - 0.5 * np.abs(ul + ur) * (ur - ul))
    f -= D * (ug[2:n + 3] - ug[1:n + 2]) * dx_inv
    return f


def solve(u0, nu, t_out, cfg: SimConfig = SimConfig(), log: CallLog | None = None,
          max_steps: int = 2_000_000):
    """Integrate from t=0 and return u at each time in t_out (shape [len(t_out), nx]).

    Raises FloatingPointError on non-finite state or step overflow (logged as a failure).
    """
    t0 = time.perf_counter()
    u = np.array(u0, dtype=np.float64)
    D = nu / np.pi
    dx = cfg.dx
    dx_inv = 1.0 / dx
    dt_dif = cfg.cfl * 0.5 * dx ** 2 / (D + 1e-8)
    out = np.empty((len(t_out), u.size))
    t, steps, ok = 0.0, 0, True
    try:
        for j, t_target in enumerate(t_out):
            while t < t_target - 1e-12:
                dt = min(cfg.cfl * dx / (np.max(np.abs(u)) + 1e-8), dt_dif, t_target - t)
                u_half = u - 0.5 * dt * dx_inv * np.diff(_flux(u, D, dx_inv, cfg.second_order))
                u = u - dt * dx_inv * np.diff(_flux(u_half, D, dx_inv, cfg.second_order))
                t += dt
                steps += 1
                if steps > max_steps:
                    raise FloatingPointError("max_steps exceeded")
            if not np.all(np.isfinite(u)):
                raise FloatingPointError("non-finite state")
            out[j] = u
    except FloatingPointError:
        ok = False
        raise
    finally:
        if log is not None:
            log.add(nu, steps, time.perf_counter() - t0, ok)
    return out


def interp_periodic(u, x_query, cfg: SimConfig):
    """Periodic linear interpolation of cell-centre values (works on [..., nx])."""
    s = (np.asarray(x_query) - cfg.domain[0]) / cfg.dx - 0.5
    i0 = np.floor(s).astype(int)
    w = s - i0
    i0 %= cfg.nx
    i1 = (i0 + 1) % cfg.nx
    return (1 - w) * u[..., i0] + w * u[..., i1]
