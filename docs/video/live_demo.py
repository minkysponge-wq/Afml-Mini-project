"""Live demo of the project, for a screen recording. Runs the real code step by step.

Usage (from the repository root):  python docs/video/live_demo.py
  --no-pause   run straight through (for testing)
  --workers N  processes for the mini benchmark (default: CPU count - 1)
Needs the Phase B dataset: python -m blinit.experiments.build_dataset --config configs/phaseB_v2.json
Each step waits for Enter, so the presenter can explain what is on screen.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")
os.system("")                       # enables ANSI colours in the Windows console
sys.stdout.reconfigure(line_buffering=True)

import numpy as np                                                    # noqa: E402

from blinit.data import features                                      # noqa: E402
from blinit.experiments.train_mlp import load_split                   # noqa: E402
from blinit.inverse import Problem, run_method                        # noqa: E402
from blinit.model import load_model, predict_theta                    # noqa: E402
from blinit.observe import ObsConfig                                   # noqa: E402
from blinit.simulator import SimConfig                                 # noqa: E402

US = 144.3e-6                       # CPU-s per solver step: the cost unit used in every result
B, DIM, CY, GR, YE, MA, RS = "\033[1m", "\033[2m", "\033[36m", "\033[32m", "\033[33m", "\033[35m", "\033[0m"
PAUSE = True
D = ROOT / "data" / "phaseB_v2"
LSQ = dict(xtol=1e-3, ftol=1e-10, diff_step=1e-6, max_nfev=100)
BRENT = dict(xatol=1e-3, maxiter=100)


def step(n, title, what):
    if PAUSE:
        input(f"\n{DIM}[press Enter for step {n}]{RS}")
    print(f"\n{CY}{B}{'=' * 78}\n STEP {n}: {title}\n{'=' * 78}{RS}")
    print(f"{DIM}{what}{RS}\n")
    time.sleep(0.4)


def show(path):
    """Open a figure in the default image viewer (Windows) so it can be shown on screen."""
    p = ROOT / path
    print(f"{GR}opening {path}{RS}")
    if PAUSE and hasattr(os, "startfile"):
        os.startfile(p)                      # noqa: S606 (local, repository file)


def make_problem(T, O, v):
    p = int(O["parent"][v])
    y = np.where(np.isnan(O["y"][v]), 0.0, O["y"][v])
    sim, obs = SimConfig.from_json(), ObsConfig.from_json()
    return lambda: Problem(T["amps"][p], T["phases"][p], y, O["mask"][v], sim, obs)


class LiveProblem(Problem):
    """Problem that prints every simulator run as it happens (for the demo only)."""
    def residual(self, theta):
        n0 = self.log.n_calls
        r = super().residual(theta)
        if self.log.n_calls > n0:
            th, J, steps = self.trace[-1]
            print(f"   run {self.log.n_calls:2d}: trying nu = {np.exp(th):.5f}   misfit = {J:.2e}   "
                  f"cost so far = {steps * US:5.2f} CPU-s", flush=True)
            time.sleep(0.15)
        return r


def mini_task(args):
    T, O, v, th_mlp = args
    mk = make_problem(T, O, v)
    out = {}
    for name, f in [("Brent", lambda: run_method("brent", mk, brent_kw=BRENT)),
                    ("MLP + LSQ", lambda: run_method("mlp+lsq", mk, init_theta=th_mlp, refine_kw=LSQ)),
                    ("MLP bracket", lambda: run_method("mlp+brent_bracket", mk, init_theta=th_mlp,
                                                       bracket_width=0.5, brent_kw=BRENT))]:
        r = f()
        out[name] = (r["n_solves"], r["solver_steps"] * US, r["nu_hat"])
    return v, out


def main():
    global PAUSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pause", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    PAUSE = not a.no_pause

    print(f"""{B}{MA}
  Does a Learned Starting Point Make Viscosity Estimation Cheaper?
  A cost-counted study on the Burgers equation{RS}
  UE24AM342AA1 Advanced Foundations in Machine Learning, PES University
  PES1UG24AM246 Sampreeth Jaladi | PES1UG24AM243 Samarth
  PES1UG24AM255 Sarva Sandip     | PES1UG24AM239 Sai Likhith S""")

    # 1 repository
    step(1, "The repository", "Solver, inverse methods, dataset builder, MLP, experiments, tests, results.")
    for d, desc in [("blinit/simulator.py", "Burgers finite-volume solver (PDEBench scheme, our own code)"),
                    ("blinit/inverse.py", "Brent search, least squares, learned bracket; counts every run"),
                    ("blinit/data.py", "dataset generation, split by seed, 9 integrity checks"),
                    ("blinit/model.py", "2x64 MLP that predicts log viscosity"),
                    ("blinit/experiments/", "one script per phase (evaluate, analyze, ...)"),
                    ("configs/", "every experiment is a versioned JSON config"),
                    ("docs/preregistration_*.md", "success rules written before each test"),
                    ("results/", "every number in the paper comes from a file here"),
                    ("tests/", "pytest suite")]:
        print(f"  {YE}{d:28s}{RS} {desc}")

    # 2 tests
    step(2, "Running the test suite",
         "Checks the solver against an exact solution (Cole-Hopf), conservation and symmetry, plus data-leak tests.")
    print(f"{GR}> python -m pytest -q{RS}")
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings"], cwd=ROOT)

    # 3 dataset
    step(3, "The dataset (generated by our solver)",
         "3,000 / 50 / 50 trajectories, each observed 9 ways: 4, 8 or 16 sensors x 0, 1 or 5% noise.")
    for split in ("train", "val", "test"):
        T, O = load_split(D, split)
        print(f"  {split:5s}: {len(T['nu']):5d} trajectories -> {len(O['parent']):6d} cases   "
              f"viscosity from {T['nu'].min():.4f} to {T['nu'].max():.4f}")
    print(f"  each case: {O['y'].shape[1]} times x {O['y'].shape[2]} sensor slots")
    integ = json.loads((D / "integrity.json").read_text())
    print(f"\n  integrity checks:")
    for k, v in integ["checks"].items():
        print(f"    {GR if v else YE}{'PASS' if v else 'FAIL'}{RS}  {k}")
    show("docs/report/figures/trajectories.png")

    # 4 MLP
    step(4, "The trained MLP makes a first guess",
         "The network reads the sensor readings and the known starting wave and guesses the viscosity in under a millisecond.")
    norm = dict(np.load(D / "normaliser.npz")); norm["fitted_on"] = str(norm["fitted_on"])
    T, O = load_split(D, "test")
    X, y = features(O, T, norm)
    model = load_model(ROOT / "models" / "phaseC_v2" / "seed0" / "model.pkl")
    t0 = time.perf_counter(); th = predict_theta(model, X); dt = time.perf_counter() - t0
    t1 = time.perf_counter(); predict_theta(model, X[:1]); d1 = time.perf_counter() - t1
    print(f"  all {len(X)} test cases in {dt * 1e3:.1f} ms; a single case in {d1 * 1e3:.2f} ms "
          f"{DIM}(one simulator run takes 0.1-1 s){RS}\n")
    print(f"  {'case':>5s}  {'sensors':>7s}  {'noise':>5s}  {'true nu':>8s}  {'MLP guess':>9s}  {'error':>6s}")
    for v in (4, 13, 85, 200, 301, 445):
        t, g = np.exp(y[v]), np.exp(th[v])
        print(f"  {v:5d}  {O['n_sensors'][v]:7d}  {O['noise'][v]:5.0%}  {t:8.5f}  {g:9.5f}  {abs(g - t) / t:6.1%}")
    e = np.abs(np.exp(th) - np.exp(y)) / np.exp(y)
    print(f"\n  median error over all {len(X)} test cases: {np.median(e):.1%}  "
          f"{DIM}(good, but far from the 0.2% the data allow: it needs refining){RS}")

    # 5 live fit
    v = 4
    step(5, f"Live fit of one test case (case {v}: {O['n_sensors'][v]} sensors, {O['noise'][v]:.0%} noise)",
         "Every line is one real simulator run. Cost is counted in CPU-seconds.")
    nu = float(T["nu"][O["parent"][v]])
    print(f"  true viscosity (hidden from the methods): {nu:.5f}")
    print(f"  MLP guess: {np.exp(th[v]):.5f}\n")
    p = int(O["parent"][v]); yv = np.where(np.isnan(O["y"][v]), 0.0, O["y"][v])
    mk = lambda: LiveProblem(T["amps"][p], T["phases"][p], yv, O["mask"][v], SimConfig.from_json(), ObsConfig.from_json())
    res = {}
    for name, f in [("Brent search (no guess, whole range)", lambda: run_method("brent", mk, brent_kw=BRENT)),
                    ("MLP guess + least squares", lambda: run_method("mlp+lsq", mk, init_theta=float(th[v]), refine_kw=LSQ)),
                    ("MLP guess as a bracket for Brent", lambda: run_method("mlp+brent_bracket", mk, init_theta=float(th[v]),
                                                                            bracket_width=0.5, brent_kw=BRENT))]:
        print(f"{B}{YE}-- {name}{RS}")
        r = f(); res[name] = r
        print(f"   {GR}=> estimate {r['nu_hat']:.5f} (error {abs(r['nu_hat'] - nu) / nu:.2%}), "
              f"{r['n_solves']} runs, {r['solver_steps'] * US:.2f} CPU-s{RS}\n")
    print(f"  {DIM}All three reach the same answer; they differ only in cost. This is one case; averages next.{RS}")

    # 6 mini benchmark
    step(6, "Mini benchmark: 18 test cases, run live in parallel",
         "Same three methods on 18 cases (2 per sensor/noise condition). The full test uses 450 cases (~45 min).")
    rng = np.random.default_rng(0)
    pick = []
    for s in (4, 8, 16):
        for nl in (0.0, 0.01, 0.05):
            idx = np.where((O["n_sensors"] == s) & (O["noise"] == nl))[0]
            pick += sorted(rng.choice(idx, 2, replace=False).tolist())
    t0 = time.time(); rows = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, (vv, out) in enumerate(ex.map(mini_task, [(T, O, vv, float(th[vv])) for vv in pick]), 1):
            rows[vv] = out
            b, m, br = out["Brent"], out["MLP + LSQ"], out["MLP bracket"]
            print(f"  [{k:2d}/{len(pick)}] case {vv:3d} ({O['n_sensors'][vv]:2d} sensors, {O['noise'][vv]:3.0%}): "
                  f"Brent {b[1]:5.2f}   MLP+LSQ {m[1]:5.2f}   bracket {br[1]:5.2f} CPU-s", flush=True)
    print(f"\n  finished in {time.time() - t0:.0f} s on {a.workers} processes\n")
    for name in ("Brent", "MLP + LSQ", "MLP bracket"):
        c = np.array([rows[vv][name][1] for vv in pick]); n = np.array([rows[vv][name][0] for vv in pick])
        print(f"  {name:12s} mean {c.mean():5.2f} CPU-s   median runs {np.median(n):4.1f}")
    print(f"  {DIM}18 cases are too few for a conclusion; the next step shows the pre-registered 450-case result.{RS}")

    # 7 results
    step(7, "Results on the full pre-registered test (450 cases, 3 MLP seeds)",
         "Read from results/*/analysis.json, the files the paper is built from.")
    for label, d in [("primary (LSQ xtol 1e-3)", "phaseD_v1_3seeds_test"),
                     ("accuracy-matched (LSQ xtol 1e-4)", "phaseD_v1_tightlsq_3seeds_test")]:
        an = json.loads((ROOT / "results" / d / "analysis.json").read_text())
        M = an["methods"]
        print(f"{B}  {label}{RS}")
        for m in ("brent", "midpoint+lsq", "grid5+lsq", "knn+lsq", "mlp_s0+lsq"):
            print(f"    {m:14s} error median {M[m]['rel_err_median']:.2%}   CPU-s mean {M[m]['cpu_s_mean']:5.2f}  median {M[m]['cpu_s_median']:5.2f}")
        for s in range(3):
            q = an["paired"][f"mlp_s{s}+lsq - brent | all"]
            ok = q["ci95"][1] < 0
            print(f"    MLP seed {s} minus Brent: {q['median_cpu_s']:+.3f} CPU-s  95% CI [{q['ci95'][0]:+.3f}, {q['ci95'][1]:+.3f}]  "
                  f"{GR + 'saving' if ok else YE + 'no clear saving'}{RS}")
        print()
    f = json.loads((ROOT / "results" / "followup_bracket_w0.5_3seeds_test" / "analysis.json").read_text())
    print(f"{B}  follow-up: MLP bracket for Brent{RS}")
    for s in range(3):
        q = f["paired"][f"mlp_s{s}_w0.5+brent_bracket - brent | all"]
        print(f"    seed {s}: {q['median_cpu_s']:+.3f} CPU-s  95% CI [{q['ci95'][0]:+.3f}, {q['ci95'][1]:+.3f}]")
    print(f"\n  {B}Verdict:{RS} the learned start helps least squares, but does not reliably beat Brent;")
    print(f"  pre-registered result not positive, 2x target not met. The bracket saves 8-10% (post hoc).")
    show("results/report/figures/test_paired_vs_brent.png")

    # 8 robustness
    step(8, "Robustness and the accuracy limit", "Shifted test sets; estimates already at the Cramer-Rao bound.")
    print("  higher-frequency starting waves : MLP error triples, saving gone")
    print("  finer-grid data (model mismatch): saving holds")
    print("  unseen sensor patterns          : MLP no better than the midpoint, saving gone")
    print("  Cramer-Rao bound                : both methods attain it (spread 0.947 Brent, 0.944 MLP)")
    show("docs/report/figures/robustness.png")

    print(f"\n{B}{MA}  Thank you.  Code, data configs, results and paper:"
          f" github.com/minkysponge-wq/Afml-Mini-project{RS}\n")
    if PAUSE:
        input(f"{DIM}[press Enter to close]{RS}")


if __name__ == "__main__":
    main()
