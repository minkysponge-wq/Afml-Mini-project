"""Phase B dataset: trajectories (split by IC seed) and observation variants.

Trajectories and observations are generated separately. Every observation variant stores the
index of its parent trajectory, and variants are only ever created inside the parent's split.
"""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .inverse import TH_HI, TH_LO
from .observe import ObsConfig, nested_mask
from .simulator import CONFIG_DIR, SimConfig, initial_condition, interp_periodic, solve

SPLITS = ("train", "val", "test")


def git_version():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).parent).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def sample_trajectory_params(base_seed, ic_seed):
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, ic_seed]))
    nu = float(np.exp(rng.uniform(TH_LO, TH_HI)))
    return nu, rng.uniform(0, 1, 3), rng.uniform(0, 2 * np.pi, 3)


def _simulate(args):
    nu, amps, phases = args[:3]
    modes = args[3] if len(args) > 3 else (1, 2, 3)
    nx = args[4] if len(args) > 4 else None
    sim = SimConfig.from_json() if nx is None else SimConfig.from_json(nx=int(nx))
    obs = ObsConfig.from_json()
    traj = solve(initial_condition(sim.xc, amps, phases, modes=modes, length=sim.length), nu, obs.times, sim)
    return traj, interp_periodic(traj, obs.sensor_x, sim)


def generate_trajectories(cfg, split, workers=8, modes=(1, 2, 3), nx=None):
    """modes: sine modes of the IC; nx: grid cells used to simulate the data (None = the simulator config)."""
    n, off = cfg["splits"][split], cfg["ic_seed_offset"][split]
    ic_seeds = off + np.arange(n)
    modes = tuple(int(m) for m in modes)
    params = [sample_trajectory_params(cfg["base_seed"], int(s)) + (modes, nx) for s in ic_seeds]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_simulate, params, chunksize=4))
    obs = ObsConfig.from_json()
    return dict(split=split, ic_seed=ic_seeds, nu=np.array([p[0] for p in params]),
                amps=np.array([p[1] for p in params]), phases=np.array([p[2] for p in params]),
                u_at_obs_times=np.array([r[0] for r in res]), y_clean=np.array([r[1] for r in res]),
                sensor_x=obs.sensor_x, times=np.array(obs.times), modes=np.array(modes),
                sim_nx=np.array(SimConfig.from_json().nx if nx is None else int(nx)),
                sim_config=(CONFIG_DIR / "simulator_v1.json").read_text(), solver_version=git_version())


def make_observations(T, cfg, levels=None, sensors=None):
    """All (n_sensors x noise) variants of every trajectory in one split."""
    ov = cfg["observation_variants"]
    levels = ov["noise_levels"] if levels is None else levels
    sensors = ov["n_sensors"] if sensors is None else sensors
    parent, y, mask, noise, nsens = [], [], [], [], []
    for j in range(len(T["nu"])):
        yc = T["y_clean"][j]
        rms = np.sqrt(np.mean(yc ** 2))
        for li, lv in enumerate(levels):
            rng = np.random.default_rng(np.random.SeedSequence([cfg["base_seed"], int(T["ic_seed"][j]), 99, li]))
            yn = yc + lv * rms * rng.standard_normal(yc.shape)   # one draw per level, shared by sensor subsets
            for s in sensors:
                m = nested_mask(s, yc.shape[1])
                parent.append(j); mask.append(m); noise.append(lv); nsens.append(s)
                y.append(np.where(m[None, :], yn, np.nan))      # unavailable readings are NaN in raw storage
    return dict(parent=np.array(parent), y=np.array(y), mask=np.array(mask), noise=np.array(noise),
                n_sensors=np.array(nsens), split=T["split"])


def named_masks(spec, n_slots=16):
    """Sensor masks from a versioned config spec: {name: {"on": [slot indices]}} -> [(name, bool mask)]."""
    out = []
    for name, d in spec.items():
        m = np.zeros(n_slots, bool)
        m[np.asarray(d["on"], int)] = True
        out.append((name, m))
    return out


def make_observations_masks(T, cfg, masks, levels=None):
    """Like make_observations, but for a list of (name, mask) pairs instead of nested sensor counts.

    The noise draw per (trajectory, level) is the same as in make_observations and is shared by all masks.
    """
    levels = cfg["observation_variants"]["noise_levels"] if levels is None else levels
    parent, y, mask, noise, nsens, names = [], [], [], [], [], []
    for j in range(len(T["nu"])):
        yc = T["y_clean"][j]
        rms = np.sqrt(np.mean(yc ** 2))
        for li, lv in enumerate(levels):
            rng = np.random.default_rng(np.random.SeedSequence([cfg["base_seed"], int(T["ic_seed"][j]), 99, li]))
            yn = yc + lv * rms * rng.standard_normal(yc.shape)
            for name, m in masks:
                m = np.asarray(m, bool)
                parent.append(j); mask.append(m); noise.append(lv); nsens.append(int(m.sum())); names.append(name)
                y.append(np.where(m[None, :], yn, np.nan))
    return dict(parent=np.array(parent), y=np.array(y), mask=np.array(mask), noise=np.array(noise),
                n_sensors=np.array(nsens), mask_name=np.array(names), split=T["split"])


def fit_normaliser(O_train):
    """Per-time mean/std over available TRAIN readings only."""
    y = O_train["y"]
    mean = np.nanmean(y, axis=(0, 2))
    std = np.nanstd(y, axis=(0, 2))
    return dict(mean=mean, std=std, fitted_on="train")


def features(O, T, norm):
    """[standardised readings (missing -> 0), mask (repeated per time), IC a*cos(phi), a*sin(phi)]."""
    z = (O["y"] - norm["mean"][None, :, None]) / norm["std"][None, :, None]
    z = np.where(np.isnan(z), 0.0, z)
    n, nt, ns = z.shape
    m = np.repeat(O["mask"][:, None, :], nt, axis=1).astype(float)
    a, ph = T["amps"][O["parent"]], T["phases"][O["parent"]]
    X = np.concatenate([z.reshape(n, -1), m.reshape(n, -1), a * np.cos(ph), a * np.sin(ph)], axis=1)
    target = np.log(T["nu"][O["parent"]])
    return X, target


def integrity_report(Ts, Os, norm):
    """Automated leakage / integrity checks; returns dict of check -> bool."""
    seeds = {s: set(Ts[s]["ic_seed"].tolist()) for s in SPLITS}
    chk = {}
    chk["ic_seeds_disjoint"] = all(not (seeds[a] & seeds[b]) for a in SPLITS for b in SPLITS if a < b)
    chk["ic_seeds_unique_within_split"] = all(len(seeds[s]) == len(Ts[s]["ic_seed"]) for s in SPLITS)
    chk["variants_parent_in_split"] = all(O["split"] == s and O["parent"].max() < len(Ts[s]["nu"])
                                          and O["parent"].min() >= 0 for s, O in Os.items())
    chk["normaliser_fitted_on_train_only"] = norm["fitted_on"] == "train"
    chk["nan_iff_masked"] = all(np.array_equal(np.isnan(O["y"]), ~np.repeat(O["mask"][:, None, :],
                                O["y"].shape[1], axis=1)) for O in Os.values())
    chk["nu_within_bounds"] = all(Ts[s]["nu"].min() >= np.exp(TH_LO) - 1e-12 and
                                  Ts[s]["nu"].max() <= np.exp(TH_HI) + 1e-12 for s in SPLITS)
    chk["all_finite_clean"] = all(np.all(np.isfinite(Ts[s]["y_clean"])) for s in SPLITS)
    # reading <-> (x, t) mapping: clean sensor values reproduce from stored trajectories
    sim = SimConfig.from_json()
    T0 = Ts["train"]
    chk["sensor_mapping_reproducible"] = bool(np.allclose(
        interp_periodic(T0["u_at_obs_times"][:5], T0["sensor_x"], sim), T0["y_clean"][:5]))
    return chk


def params_reproducible(T, base_seed, k=5):
    """Trajectory parameters regenerate exactly from (base_seed, ic_seed)."""
    for j in range(min(k, len(T["nu"]))):
        nu, a, ph = sample_trajectory_params(base_seed, int(T["ic_seed"][j]))[:3]
        if not (nu == T["nu"][j] and np.array_equal(a, T["amps"][j]) and np.array_equal(ph, T["phases"][j])):
            return False
    return True


def load_config(path=CONFIG_DIR / "phaseB_v1.json"):
    return json.loads(Path(path).read_text())
