"""T2: accuracy-versus-cost curves across stopping tolerances (validation).

Usage: python -m blinit.experiments.tolerance_curves
Reads the per-tolerance result folders listed in POINTS, measures each estimate's precision against the
anchor run (Brent, xatol 1e-7: the per-case reference nu_ref, not a method) and writes
results/phaseF_tolerance/{curves.csv, matched.json, tolerance_curves.png}.

cost      = mean / median CPU-s per query (solver steps x us_per_step)
precision = p90 / max of |nu_hat - nu_ref| / nu_ref over the 450 validation variants
The cost ratio Brent / MLP at matched precision is read by log-log interpolation along both curves.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANCHOR = "phaseF_tol_anchor_val"
# (method, tolerance, result folder)
POINTS = [
    ("brent", 1e-2, "phaseF_tol_brent1e-2_val"),
    ("brent", 1e-3, "phaseD_v1_val"),
    ("brent", 1e-4, "phaseF_tol_brent1e-4_val"),
    ("midpoint+lsq", 1e-2, "phaseF_tol_lsq1e-2_val"),
    ("midpoint+lsq", 1e-3, "phaseD_v1_val"),
    ("midpoint+lsq", 1e-4, "phaseD_v1_tightlsq_val"),
    ("midpoint+lsq", 1e-5, "phaseF_tol_lsq1e-5_val"),
    ("mlp_s0+lsq", 1e-2, "phaseF_tol_lsq1e-2_val"),
    ("mlp_s0+lsq", 1e-3, "phaseD_v1_val"),
    ("mlp_s0+lsq", 1e-4, "phaseD_v1_tightlsq_val"),
    ("mlp_s0+lsq", 1e-5, "phaseF_tol_lsq1e-5_val"),
]
COLORS = {"brent": "#2a78d6", "midpoint+lsq": "#eb6834", "mlp_s0+lsq": "#1baf7a"}


def load(folder, method):
    out = {}
    with open(ROOT / "results" / folder / "runs.csv") as f:
        for r in csv.DictReader(f):
            if r["method"] == method:
                out[int(float(r["variant"]))] = (float(r["nu_hat"]), float(r["solver_steps"]), float(r["n_solves"]))
    return out


def interp_cost(curve, target):
    """Cost at a target precision by log-log interpolation; None outside the curve's precision range."""
    pts = sorted(curve)                      # ascending precision error
    p, c = np.log([a for a, _ in pts]), np.log([b for _, b in pts])
    if not (p[0] <= np.log(target) <= p[-1]):
        return None
    return float(np.exp(np.interp(np.log(target), p, c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--us-per-step", type=float, default=144.3)
    args = ap.parse_args()
    out = ROOT / "results" / "phaseF_tolerance"
    out.mkdir(parents=True, exist_ok=True)
    ref = {v: x[0] for v, x in load(ANCHOR, "brent").items()}
    rows = []
    for method, tol, folder in POINTS:
        d = load(folder, method)
        assert set(d) == set(ref), (method, folder, len(d), len(ref))
        vs = sorted(ref)
        prec = np.array([abs(d[v][0] - ref[v]) / ref[v] for v in vs])
        cpu = np.array([d[v][1] for v in vs]) * args.us_per_step * 1e-6
        solves = np.array([d[v][2] for v in vs])
        rows.append(dict(method=method, tolerance=tol, folder=folder, n=len(vs),
                         cpu_s_mean=float(cpu.mean()), cpu_s_median=float(np.median(cpu)),
                         solves_mean=float(solves.mean()), solves_median=float(np.median(solves)),
                         prec_median=float(np.median(prec)), prec_p90=float(np.percentile(prec, 90)),
                         prec_max=float(prec.max())))
    with open(out / "curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    matched = {}
    for stat in ("prec_p90", "prec_max"):
        for cost in ("cpu_s_mean", "cpu_s_median"):
            curves = {m: [(r[stat], r[cost]) for r in rows if r["method"] == m] for m in COLORS}
            lo = max(min(p for p, _ in c) for c in curves.values())
            hi = min(max(p for p, _ in c) for c in curves.values())
            res = dict(overlap=[lo, hi], targets={})
            for target in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
                cs = {m: interp_cost(c, target) for m, c in curves.items()}
                entry = dict(cost=cs)
                if cs["brent"] and cs["mlp_s0+lsq"]:
                    entry["brent_over_mlp"] = cs["brent"] / cs["mlp_s0+lsq"]
                if cs["midpoint+lsq"] and cs["mlp_s0+lsq"]:
                    entry["midpoint_over_mlp"] = cs["midpoint+lsq"] / cs["mlp_s0+lsq"]
                res["targets"][f"{target:g}"] = entry
            matched[f"{cost} at matched {stat}"] = res
    (out / "matched.json").write_text(json.dumps(matched, indent=2))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, stat, lab in zip(axs, ("prec_p90", "prec_max"), ("p90", "max")):
        for m, col in COLORS.items():
            rs = [r for r in rows if r["method"] == m]
            ax.plot([r["cpu_s_mean"] for r in rs], [r[stat] for r in rs], "o-", color=col, label=m, ms=5)
            for r in rs:
                ax.annotate(f"{r['tolerance']:.0e}", (r["cpu_s_mean"], r[stat]), fontsize=6,
                            textcoords="offset points", xytext=(4, 3), color=col)
        ax.set_yscale("log"); ax.set_xlabel("mean CPU-s per query")
        ax.set_ylabel(f"{lab} precision error  |nu_hat - nu_ref| / nu_ref")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Accuracy versus cost across stopping tolerances (validation, 450 variants; labels = tolerance)")
    fig.tight_layout(); fig.savefig(out / "tolerance_curves.png", dpi=150); plt.close(fig)

    for r in rows:
        print(f"{r['method']:13s} tol {r['tolerance']:.0e} | CPU-s mean {r['cpu_s_mean']:6.2f} med {r['cpu_s_median']:6.2f} "
              f"| solves mean {r['solves_mean']:5.2f} | precision med {r['prec_median']:.1e} p90 {r['prec_p90']:.1e} max {r['prec_max']:.1e}")
    print(json.dumps(matched, indent=1))


if __name__ == "__main__":
    main()
