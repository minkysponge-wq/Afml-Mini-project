"""Phase F (T4): build one held-out robustness test set and run its integrity checks.

Usage: python -m blinit.experiments.build_shift_set --config configs/phaseF_highfreq.json [--workers 8]
Outputs: data/<config_version>/{traj_test.npz, obs_test.npz, integrity.json}

A shift set has a test split only. Train, validation, the normaliser, the kNN library and the models stay
those of the base dataset (config key "base_dataset"); nothing here is used for training or tuning.
Config keys beyond phaseB: ic.modes (sine modes of the IC), sim_nx (grid cells used to simulate the data;
null = the simulator config), masks (optional {name: {"on": [slots]}}: named sensor masks instead of the
nested 4/8/16 patterns).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from blinit.data import (generate_trajectories, load_config, make_observations, make_observations_masks,
                         named_masks, params_reproducible)
from blinit.inverse import TH_HI, TH_LO
from blinit.simulator import SimConfig, interp_periodic

ROOT = Path(__file__).resolve().parents[2]


def shift_integrity(T, O, cfg, base_seeds):
    seeds = T["ic_seed"].tolist()
    sim = SimConfig.from_json(nx=int(T["sim_nx"]))
    chk = {}
    chk["ic_seeds_disjoint_from_base_dataset"] = not (set(seeds) & base_seeds)
    chk["ic_seeds_unique"] = len(set(seeds)) == len(seeds)
    chk["variants_parent_in_range"] = bool(O["parent"].min() >= 0 and O["parent"].max() < len(T["nu"]))
    chk["nan_iff_masked"] = bool(np.array_equal(np.isnan(O["y"]),
                                                ~np.repeat(O["mask"][:, None, :], O["y"].shape[1], axis=1)))
    chk["nu_within_bounds"] = bool(T["nu"].min() >= np.exp(TH_LO) - 1e-12 and T["nu"].max() <= np.exp(TH_HI) + 1e-12)
    chk["all_finite_clean"] = bool(np.all(np.isfinite(T["y_clean"])))
    chk["sensor_mapping_reproducible"] = bool(np.allclose(
        interp_periodic(T["u_at_obs_times"][:5], T["sensor_x"], sim), T["y_clean"][:5]))
    chk["params_reproducible"] = bool(params_reproducible(T, cfg["base_seed"]))
    clean = O["noise"] == 0
    chk["noise_free_equal_clean_where_available"] = bool(np.array_equal(
        O["y"][clean][O["mask"][clean][:, None, :].repeat(O["y"].shape[1], axis=1)],
        T["y_clean"][O["parent"][clean]][O["mask"][clean][:, None, :].repeat(O["y"].shape[1], axis=1)]))
    return chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert list(cfg["splits"]) == ["test"], "a shift set has a test split only"
    out = ROOT / "data" / cfg["config_version"]
    out.mkdir(parents=True, exist_ok=True)
    base = ROOT / "data" / cfg["base_dataset"]
    base_seeds = set()
    for s in ("train", "val", "test"):
        base_seeds |= set(np.load(base / f"traj_{s}.npz")["ic_seed"].tolist())

    t0 = time.perf_counter()
    T = generate_trajectories(cfg, "test", args.workers, modes=cfg["ic"]["modes"], nx=cfg.get("sim_nx"))
    wall = time.perf_counter() - t0
    if cfg.get("masks"):
        O = make_observations_masks(T, cfg, named_masks(cfg["masks"]))
    else:
        O = make_observations(T, cfg)
    np.savez_compressed(out / "traj_test.npz", **T)
    np.savez_compressed(out / "obs_test.npz", **O)
    chk = shift_integrity(T, O, cfg, base_seeds)
    rep = dict(checks=chk, all_passed=all(chk.values()), n_base_seeds=len(base_seeds),
               n_trajectories=len(T["nu"]), n_variants=len(O["parent"]), generation_wall_seconds=wall,
               workers=args.workers, config=cfg, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "integrity.json").write_text(json.dumps(rep, indent=2))
    print(f"{cfg['config_version']}: {len(T['nu'])} trajectories, {len(O['parent'])} variants, {wall:.0f}s wall")
    print(json.dumps(chk, indent=1))
    if not rep["all_passed"]:
        raise SystemExit("INTEGRITY CHECK FAILED")


if __name__ == "__main__":
    main()
