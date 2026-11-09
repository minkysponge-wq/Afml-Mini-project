"""Phase B: build trajectory splits + observation variants, run integrity checks.

Usage: python -m blinit.experiments.build_dataset [--config configs/phaseB_v1.json] [--workers 8]
Outputs: data/<config_version>/{traj_<split>.npz, obs_<split>.npz, normaliser.npz, integrity.json}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from blinit.data import (SPLITS, fit_normaliser, generate_trajectories, integrity_report, load_config,
                         make_observations, params_reproducible)
from blinit.simulator import CONFIG_DIR

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_DIR / "phaseB_v1.json"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ROOT / "data" / cfg["config_version"]
    out.mkdir(parents=True, exist_ok=True)
    Ts, Os, cost = {}, {}, {}
    for s in SPLITS:
        t0 = time.perf_counter()
        Ts[s] = generate_trajectories(cfg, s, args.workers)
        cost[s] = time.perf_counter() - t0
        Os[s] = make_observations(Ts[s], cfg)
        np.savez_compressed(out / f"traj_{s}.npz", **Ts[s])
        np.savez_compressed(out / f"obs_{s}.npz", **Os[s])
        print(f"{s}: {len(Ts[s]['nu'])} trajectories, {len(Os[s]['parent'])} observation variants, "
              f"{cost[s]:.0f}s wall ({args.workers} workers)", flush=True)
    norm = fit_normaliser(Os["train"])
    np.savez(out / "normaliser.npz", **norm)
    chk = integrity_report(Ts, Os, norm)
    chk["params_reproducible"] = all(params_reproducible(Ts[s], cfg["base_seed"]) for s in SPLITS)
    rep = dict(checks=chk, all_passed=all(chk.values()), generation_wall_seconds=cost,
               workers=args.workers, config=cfg, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "integrity.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(chk, indent=1))
    if not rep["all_passed"]:
        raise SystemExit("INTEGRITY CHECK FAILED")


if __name__ == "__main__":
    main()
