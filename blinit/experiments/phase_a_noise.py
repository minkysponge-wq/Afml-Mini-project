"""Phase A': classical recovery under noise and fewer sensors (development problems only).

Reuses the 20 Phase A problems (results/phaseA/problems.npz). Noise: Gaussian, sigma = level *
RMS of that trajectory's clean 16x4 observations; one draw per (problem, level) on all 16 slots,
then masked, so 4/8/16-sensor cases are nested subsets of the same noisy data.

Runs problems in parallel processes: solve counts and errors are valid; wall times are NOT
comparable (shared CPU) and are not reported from this experiment.

Usage: python -m blinit.experiments.phase_a_noise [--workers 12] [--quick]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from blinit.inverse import TH_HI, TH_LO, Problem, run_method
from blinit.observe import ObsConfig, nested_mask
from blinit.simulator import CONFIG_DIR, SimConfig

ROOT = Path(__file__).resolve().parents[2]


def noisy_obs(y_clean, level, seed, i, li):
    if level == 0:
        return y_clean.copy()
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2, i, li)))
    sigma = level * np.sqrt(np.mean(y_clean ** 2))
    return y_clean + sigma * rng.standard_normal(y_clean.shape)


def run_problem(args):
    i, P, cfg, conds, n_scan = args
    sim, obs = SimConfig.from_json(), ObsConfig.from_json()
    m = cfg["methods"]
    scan_th = np.linspace(TH_LO, TH_HI, n_scan)
    rows, scans = [], {}
    for n_sens, level, li in conds:
        y = noisy_obs(P["y_obs"][i], level, cfg["seed"], i, li)
        mask = nested_mask(n_sens, obs.n_slots)
        mk = lambda: Problem(P["amps"][i], P["phases"][i], y, mask, sim, obs)
        ps = mk()
        J = np.array([ps.objective(t) for t in scan_th])
        scans[(n_sens, level)] = J
        runs = [run_method("brent", mk, brent_kw=m["brent"]),
                run_method("midpoint+lsq", mk, refine_kw=m["lsq_refine"]),
                run_method(f"grid{m['grid_k']}+lsq", mk, refine_kw=m["lsq_refine"], grid_k=m["grid_k"]),
                run_method("ORACLE+lsq(diagnostic)", mk, init_theta=np.log(P["nu"][i]), refine_kw=m["lsq_refine"])]
        # reference optimum: best objective found by any method or the scan (no ground truth used)
        cands = [(r["objective"], r["theta_hat"]) for r in runs] + [(J.min(), scan_th[J.argmin()])]
        J_ref, th_ref = min(cands)
        n_min = int(np.sum((J[1:-1] < J[:-2]) & (J[1:-1] < J[2:])) + (J[0] < J[1]) + (J[-1] < J[-2]))
        for r in runs:
            r.pop("trace", None)
            r.update(problem=i, n_sensors=n_sens, noise=level, nu_true=float(P["nu"][i]),
                     rel_err=abs(r["nu_hat"] - P["nu"][i]) / P["nu"][i],
                     rel_err_vs_ref=abs(r["nu_hat"] - np.exp(th_ref)) / np.exp(th_ref),
                     ref_rel_err=abs(np.exp(th_ref) - P["nu"][i]) / P["nu"][i],
                     obj_gap=r["objective"] - J_ref, J_ref=J_ref, scan_local_minima=n_min,
                     scan_flatness=float(J.min() / J.max()))
            rows.append(r)
    return i, rows, scans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dir", default=str(ROOT / "results" / "phaseA"))
    args = ap.parse_args()
    d = Path(args.dir)
    out = d.parent / ("phaseA_noise_quick" if args.quick else "phaseA_noise")
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((CONFIG_DIR / "phaseA_v1.json").read_text())
    P = dict(np.load(d / "problems.npz"))
    levels = [0.0, 0.01, 0.05]
    conds = [(s, lv, li) for li, lv in enumerate(levels) for s in (4, 8, 16) if not (s == 16 and lv == 0)]
    n_prob = 2 if args.quick else len(P["nu"])
    n_scan = 9 if args.quick else 25
    if args.quick:
        conds = conds[:2]
    t0 = time.time()
    rows, scans = [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r, s in ex.map(run_problem, [(i, P, cfg, conds, n_scan) for i in range(n_prob)]):
            rows += r
            scans.update({(i,) + k: v for k, v in s.items()})
            print(f"problem {i} done ({time.time() - t0:.0f}s)", flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    sk = sorted(scans)
    np.savez(out / "scans.npz", keys=np.array(sk, dtype=float), J=np.array([scans[k] for k in sk]),
             scan_theta=np.linspace(TH_LO, TH_HI, n_scan))
    (out / "meta.json").write_text(json.dumps(dict(
        config=cfg, noise_def="sigma = level * RMS(clean 16x4 obs of that trajectory)",
        noise_seed="SeedSequence(cfg.seed, spawn_key=(2, problem, level_index))",
        conditions=[c[:2] for c in conds], n_scan=n_scan, workers=args.workers,
        timing_note="parallel processes: wall times not comparable; use solve counts",
        finished=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
