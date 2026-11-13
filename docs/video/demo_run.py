"""Live demo used in the presentation video: dataset summary, then one test case solved three ways.

Usage: python docs/video/demo_run.py [--variant 211] [--json docs/video/demo_trace.json]
Needs the Phase B dataset (python -m blinit.experiments.build_dataset --config configs/phaseB_v2.json).
Uses only repository code and the saved MLP (models/phaseC_v2/seed0). Prints every simulator call.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")   # scikit-learn version notice when unpickling

from blinit.data import features                                     # noqa: E402
from blinit.experiments.train_mlp import load_split                   # noqa: E402
from blinit.inverse import TH_HI, TH_LO, Problem, run_method          # noqa: E402
from blinit.model import load_model, predict_theta                    # noqa: E402
from blinit.observe import ObsConfig                                   # noqa: E402
from blinit.simulator import SimConfig                                 # noqa: E402

US = 144.3e-6   # CPU-s per solver step, as in every result of the project


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, default=211)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    d = ROOT / "data" / "phaseB_v2"

    print("== Dataset (data/phaseB_v2) ==")
    integ = json.loads((d / "integrity.json").read_text())
    for split in ("train", "val", "test"):
        T, O = load_split(d, split)
        print(f"{split:5s}: {len(T['nu']):5d} trajectories -> {len(O['parent']):6d} cases   "
              f"nu in [{T['nu'].min():.4f}, {T['nu'].max():.4f}]")
    print(f"readings per case: {O['y'].shape[1]} times x {O['y'].shape[2]} sensor slots; "
          f"sensors on: {sorted(set(O['n_sensors'].tolist()))}; noise: {sorted(set(O['noise'].tolist()))}")
    print("integrity checks: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in integ["checks"].items()))

    norm = dict(np.load(d / "normaliser.npz")); norm["fitted_on"] = str(norm["fitted_on"])
    T, O = load_split(d, "test")
    X, y = features(O, T, norm)
    v = a.variant
    p = int(O["parent"][v]); nu = float(T["nu"][p])
    model = load_model(ROOT / "models" / "phaseC_v2" / "seed0" / "model.pkl")
    th_mlp = float(predict_theta(model, X[v:v + 1])[0])
    print(f"\n== Test case {v}: {int(O['n_sensors'][v])} sensors, {O['noise'][v]:.0%} noise, true nu = {nu:.5f} ==")
    print(f"MLP guess: nu = {np.exp(th_mlp):.5f}  ({abs(np.exp(th_mlp) - nu) / nu:.1%} off)")

    sim, obs = SimConfig.from_json(), ObsConfig.from_json()
    y_full = np.where(np.isnan(O["y"][v]), 0.0, O["y"][v])
    mk = lambda: Problem(T["amps"][p], T["phases"][p], y_full, O["mask"][v], sim, obs)
    out = dict(variant=v, nu_true=nu, n_sensors=int(O["n_sensors"][v]), noise=float(O["noise"][v]),
               theta_mlp=th_mlp, methods={})
    ps = mk()
    scan = np.linspace(TH_LO, TH_HI, 41)
    out["scan"] = [[float(t), ps.objective(t)] for t in scan]   # for the figure only; not part of any method's cost

    runs = [("Brent (full range)", lambda: run_method("brent", mk, brent_kw=dict(xatol=1e-3, maxiter=100))),
            ("MLP start + least squares", lambda: run_method("mlp+lsq", mk, init_theta=th_mlp,
                                                             refine_kw=dict(xtol=1e-3, ftol=1e-10, diff_step=1e-6, max_nfev=100))),
            ("MLP bracket + Brent", lambda: run_method("mlp+brent_bracket", mk, init_theta=th_mlp, bracket_width=0.5,
                                                       brent_kw=dict(xatol=1e-3, maxiter=100)))]
    for name, f in runs:
        r = f()
        tr = json.loads(r["trace"])
        print(f"\n-- {name} --")
        for k, (th, J, steps) in enumerate(tr, 1):
            print(f"  run {k:2d}: nu = {np.exp(th):.5f}   misfit J = {J:.3e}   cost so far = {steps * US:5.2f} CPU-s")
        print(f"  => nu_hat = {r['nu_hat']:.5f}  error {abs(r['nu_hat'] - nu) / nu:.2%}  |  "
              f"{r['n_solves']} simulator runs, {r['solver_steps'] * US:.2f} CPU-s")
        out["methods"][name] = dict(trace=tr, nu_hat=r["nu_hat"], n_solves=r["n_solves"],
                                    cpu_s=r["solver_steps"] * US, rel_err=abs(r["nu_hat"] - nu) / nu)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
