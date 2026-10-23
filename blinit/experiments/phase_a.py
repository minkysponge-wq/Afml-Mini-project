"""Phase A: simulator validation and classical recovery pilot (noise-free, 16 sensors).

Usage:  python -m blinit.experiments.phase_a [--n 20] [--quick]
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from blinit.inverse import TH_HI, TH_LO, Problem, run_method
from blinit.observe import ObsConfig, nested_mask, observe
from blinit.simulator import CONFIG_DIR, CallLog, SimConfig, initial_condition, interp_periodic, solve

ROOT = Path(__file__).resolve().parents[2]


def power_status():
    """Best-effort AC/battery state (timing depends on it on laptops)."""
    try:
        import subprocess
        r = subprocess.run(["powershell", "-NoProfile", "-c", "(Get-CimInstance Win32_Battery).BatteryStatus"],
                           capture_output=True, text=True, timeout=20)
        return {"win32_battery_status": r.stdout.strip(), "note": "2 = on AC power"}
    except Exception as e:  # non-Windows or no battery
        return {"error": repr(e)}


def sample_problems(n, seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0,)))
    th = rng.uniform(TH_LO, TH_HI, n)
    amps = rng.uniform(0, 1, (n, 3))
    phases = rng.uniform(0, 2 * np.pi, (n, 3))
    return np.exp(th), amps, phases


def grid_convergence(cfg, obs):
    gc = cfg["grid_convergence"]
    rng = np.random.default_rng(np.random.SeedSequence(cfg["seed"], spawn_key=(1,)))
    rows = []
    for i in range(gc["n_ics"]):
        a, ph = rng.uniform(0, 1, 3), rng.uniform(0, 2 * np.pi, 3)
        for nu in gc["nus"]:
            ref = observe(a, ph, nu, SimConfig(nx=gc["nx_ref"]), obs)
            # sensitivity scale: change in observations for a 1% change in nu (at nx=256)
            s256 = SimConfig(nx=256)
            sens = np.sqrt(np.mean((observe(a, ph, nu * 1.01, s256, obs) - observe(a, ph, nu, s256, obs)) ** 2))
            for nx in gc["nx_list"]:
                y = observe(a, ph, nu, SimConfig(nx=nx), obs)
                rows.append(dict(ic=i, nu=nu, nx=nx, rms_err=float(np.sqrt(np.mean((y - ref) ** 2))),
                                 max_err=float(np.max(np.abs(y - ref))), rms_sens_1pct=float(sens)))
                print(f"  gridconv ic={i} nu={nu} nx={nx} rms_err={rows[-1]['rms_err']:.2e} "
                      f"sens1%={sens:.2e}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_DIR / "phaseA_v1.json"))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (3 problems, coarse scan, no grid study)")
    ap.add_argument("--out", default=str(ROOT / "results" / "phaseA"))
    ap.add_argument("--reuse-grid", action="store_true",
                    help="copy grid-convergence rows from an existing meta.json instead of recomputing")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    n = args.n or (3 if args.quick else cfg["n_problems"])
    n_scan = 9 if args.quick else cfg["scan_points"]
    out = Path(args.out + ("_quick" if args.quick else ""))
    out.mkdir(parents=True, exist_ok=True)

    sim, obs = SimConfig.from_json(), ObsConfig.from_json()
    mask = nested_mask(cfg["n_active_sensors"], obs.n_slots)
    nus, amps, phases = sample_problems(n, cfg["seed"])
    m = cfg["methods"]

    meta = dict(config=cfg, python=sys.version, platform=platform.platform(),
                processor=platform.processor(), numpy=np.__version__,
                power=power_status(), started=time.strftime("%Y-%m-%d %H:%M:%S"), timing="sequential, single process, "
                "time.perf_counter wall clock, float64 NumPy on CPU, no batching")
    rows, scans, obs_store = [], [], []
    scan_th = np.linspace(TH_LO, TH_HI, n_scan)
    for i in range(n):
        gen_log = CallLog()
        y = observe(amps[i], phases[i], nus[i], sim, obs, gen_log)
        obs_store.append(y)
        mk = lambda: Problem(amps[i], phases[i], y, mask, sim, obs)
        # Diagnostic objective scan (cost not attributed to any method)
        ps = mk()
        scans.append([ps.objective(t) for t in scan_th])
        runs = [run_method("brent", mk, brent_kw=m["brent"]),
                run_method("midpoint+lsq", mk, refine_kw=m["lsq_refine"]),
                run_method(f"grid{m['grid_k']}+lsq", mk, refine_kw=m["lsq_refine"], grid_k=m["grid_k"]),
                run_method("ORACLE+lsq(diagnostic)", mk, init_theta=np.log(nus[i]), refine_kw=m["lsq_refine"])]
        for r in runs:
            r.update(problem=i, nu_true=float(nus[i]), abs_log_err=abs(r["theta_hat"] - np.log(nus[i])),
                     rel_err=abs(r["nu_hat"] - nus[i]) / nus[i], scan_min_objective=float(np.min(scans[-1])))
            rows.append(r)
        print(f"[{i+1}/{n}] nu={nus[i]:.4f} " + " | ".join(
            f"{r['method']}: err={r['rel_err']:.1e} solves={r['n_solves']} t={r['wall_seconds']:.1f}s"
            for r in runs), flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    np.savez(out / "problems.npz", nu=nus, amps=amps, phases=phases, y_obs=np.array(obs_store),
             mask=mask, sensor_x=obs.sensor_x, times=np.array(obs.times), scan_theta=scan_th,
             scan_objective=np.array(scans))
    if args.reuse_grid and (out / "meta.json").exists():
        meta["grid_convergence"] = json.loads((out / "meta.json").read_text())["grid_convergence"]
        meta["grid_convergence_note"] = "reused from previous run (deterministic, timing-independent)"
    elif not args.quick:
        meta["grid_convergence"] = grid_convergence(cfg, obs)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("saved to", out)


if __name__ == "__main__":
    main()
