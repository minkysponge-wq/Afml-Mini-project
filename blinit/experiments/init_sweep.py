"""DIAGNOSTIC: how much does starting-point quality reduce refinement cost?

Starts the shared least-squares refinement at theta_true + delta (delta in log-nu units) — a
truth-based, NON-deployable initialiser — and compares with Brent at several stopping tolerances.
Answers: what initialiser accuracy would be needed to beat the strongest classical baseline?

Usage: python -m blinit.experiments.init_sweep [--dir results/phaseA]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from blinit.inverse import TH_HI, TH_LO, Problem, run_method
from blinit.observe import ObsConfig
from blinit.simulator import SimConfig

DELTAS = [-0.3, -0.1, -0.03, -0.01, 0.01, 0.03, 0.1, 0.3]   # log-nu offsets (~1%..35%)
BRENT_XATOL = [1e-2, 1e-3, 1e-4]
LSQ_XTOL = [1e-2, 1e-3, 1e-4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/phaseA")
    d = Path(ap.parse_args().dir)
    P = np.load(d / "problems.npz")
    cfg = json.loads((Path("configs") / "phaseA_v1.json").read_text())
    sim, obs = SimConfig.from_json(), ObsConfig.from_json()
    rows = []
    for i in range(len(P["nu"])):
        th_true = np.log(P["nu"][i])
        mk = lambda: Problem(P["amps"][i], P["phases"][i], P["y_obs"][i], P["mask"], sim, obs)
        runs = []
        for xa in BRENT_XATOL:
            r = run_method("brent", mk, brent_kw=dict(xatol=xa, maxiter=100))
            r.update(setting=f"xatol={xa:g}", delta=None)
            runs.append(r)
        for xt in LSQ_XTOL:
            kw = dict(cfg["methods"]["lsq_refine"], xtol=xt)
            for dl in DELTAS:
                th0 = float(np.clip(th_true + dl, TH_LO, TH_HI))
                r = run_method("perturbed-oracle+lsq(diagnostic)", mk, init_theta=th0, refine_kw=kw)
                r.update(setting=f"xtol={xt:g}", delta=dl)
                runs.append(r)
        for r in runs:
            r.update(problem=i, nu_true=float(P["nu"][i]),
                     rel_err=abs(r["nu_hat"] - P["nu"][i]) / P["nu"][i])
            r.pop("trace", None)
        rows += runs
        print(f"[{i+1}/{len(P['nu'])}] done", flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(d / "init_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # summary table
    out = {}
    for xa in BRENT_XATOL:
        rs = [r for r in rows if r["method"] == "brent" and r["setting"] == f"xatol={xa:g}"]
        out[f"brent xatol={xa:g}"] = dict(solves_median=float(np.median([r["n_solves"] for r in rs])),
                                          rel_err_median=float(np.median([r["rel_err"] for r in rs])),
                                          rel_err_max=float(np.max([r["rel_err"] for r in rs])))
    for xt in LSQ_XTOL:
        for dl in sorted({abs(x) for x in DELTAS}):
            rs = [r for r in rows if r["delta"] is not None and abs(r["delta"]) == dl and r["setting"] == f"xtol={xt:g}"]
            out[f"lsq xtol={xt:g} |delta|={dl:g}"] = dict(
                solves_median=float(np.median([r["n_solves"] for r in rs])),
                rel_err_median=float(np.median([r["rel_err"] for r in rs])),
                rel_err_max=float(np.max([r["rel_err"] for r in rs])))
    (d / "init_sweep_summary.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(f"{k:32s} solves_med={v['solves_median']:5.1f}  err_med={v['rel_err_median']:.1e}  err_max={v['rel_err_max']:.1e}")


if __name__ == "__main__":
    main()
