"""T3: compare the estimators with the Cramer-Rao bound (noisy conditions only).

Usage:
  python -m blinit.experiments.crb --split val  [--runs-dir results/phaseD_v1_val]
  python -m blinit.experiments.crb --split test [--runs-dir results/phaseD_v1_3seeds_test]
Outputs in results/phaseF_crb/: crb_<split>.csv (one row per noisy variant and method),
table_<split>.json (condition x RMSE x sqrt(mean CRB) x std(z)) and crb_<split>.png.

sigma = noise level x RMS of the clean 16x4 readings of the parent trajectory (as in data.make_observations).
Ground truth is used here for scoring only. The bound assumes an unbiased estimator and ignores the
parameter bounds; noise-free variants have a bound of 0 and are excluded.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from blinit.crb import crb_theta, sensitivity
from blinit.experiments.train_mlp import load_split

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("brent", "mlp_s0+lsq")
COLORS = {"brent": "#2a78d6", "mlp_s0+lsq": "#1baf7a"}


def _sens(args):
    return sensitivity(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--dataset", default="phaseB_v2")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir).resolve() if args.runs_dir else ROOT / "results" / (
        "phaseD_v1_val" if args.split == "val" else "phaseD_v1_3seeds_test")
    out = ROOT / "results" / "phaseF_crb"
    out.mkdir(parents=True, exist_ok=True)
    T, O = load_split(ROOT / "data" / args.dataset, args.split)
    n_traj = len(T["nu"])
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        S = list(ex.map(_sens, [(T["amps"][j], T["phases"][j], float(T["nu"][j])) for j in range(n_traj)]))
    rms = np.sqrt(np.mean(T["y_clean"] ** 2, axis=(1, 2)))

    est = {}
    with open(runs_dir / "runs.csv") as f:
        for r in csv.DictReader(f):
            if r["method"] in METHODS:
                est[(r["method"], int(float(r["variant"])))] = (float(r["theta_hat"]), float(r["nu_true"]))

    rows = []
    for v in range(len(O["parent"])):
        lv = float(O["noise"][v])
        if lv == 0.0:
            continue
        j = int(O["parent"][v])
        bound = crb_theta(S[j], O["mask"][v], lv * rms[j])
        for m in METHODS:
            th, nu_true = est[(m, v)]
            assert abs(nu_true - float(T["nu"][j])) < 1e-12
            err = th - np.log(nu_true)
            rows.append(dict(variant=v, parent=j, n_sensors=int(O["n_sensors"][v]), noise=lv, method=m,
                             nu_true=nu_true, theta_err=err, crb_var=bound, crb_sd=np.sqrt(bound),
                             z=err / np.sqrt(bound)))
    with open(out / f"crb_{args.split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    table = []
    for s in (4, 8, 16):
        for lv in (0.01, 0.05):
            for m in METHODS:
                rs = [r for r in rows if r["n_sensors"] == s and r["noise"] == lv and r["method"] == m]
                e = np.array([r["theta_err"] for r in rs]); z = np.array([r["z"] for r in rs])
                b = np.array([r["crb_var"] for r in rs])
                table.append(dict(n_sensors=s, noise=lv, method=m, n=len(rs),
                                  rmse_theta=float(np.sqrt(np.mean(e ** 2))), sqrt_mean_crb=float(np.sqrt(b.mean())),
                                  ratio_rmse_over_bound=float(np.sqrt(np.mean(e ** 2)) / np.sqrt(b.mean())),
                                  mean_z=float(z.mean()), std_z=float(z.std(ddof=1)),
                                  median_abs_z=float(np.median(np.abs(z)))))
    allz = {m: np.array([r["z"] for r in rows if r["method"] == m]) for m in METHODS}
    summary = dict(split=args.split, runs_dir=str(runs_dir.relative_to(ROOT)).replace("\\", "/"), sensitivity_step=1e-3,
                   table=table, overall={m: dict(n=len(z), mean_z=float(z.mean()), std_z=float(z.std(ddof=1)),
                                                 frac_abs_z_gt_2=float(np.mean(np.abs(z) > 2))) for m, z in allz.items()})
    (out / f"table_{args.split}.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for m in METHODS:
        ts = [t for t in table if t["method"] == m]
        ax.scatter([t["sqrt_mean_crb"] for t in ts], [t["rmse_theta"] for t in ts], s=36,
                   color=COLORS[m], label=m, marker="o" if m == "brent" else "x")
    for t in table:
        if t["method"] == "brent":
            ax.annotate(f"{t['n_sensors']}s/{t['noise']:.0%}", (t["sqrt_mean_crb"], t["rmse_theta"]), fontsize=7,
                        textcoords="offset points", xytext=(5, -9))
    lim = [0.5 * min(t["sqrt_mean_crb"] for t in table), 2 * max(max(t["sqrt_mean_crb"], t["rmse_theta"]) for t in table)]
    ax.plot(lim, lim, color="0.5", lw=0.8, ls="--", label="y = x (bound attained)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("sqrt(mean Cramer-Rao bound) for log nu"); ax.set_ylabel("empirical RMSE of log nu")
    ax.set_title(f"Estimation error vs Cramer-Rao bound ({args.split} split)", fontsize=10)
    ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / f"crb_{args.split}.png", dpi=150); plt.close(fig)

    for t in table:
        print(f"s{t['n_sensors']:2d} n{t['noise']:.0%} {t['method']:11s} RMSE {t['rmse_theta']:.4f} "
              f"sqrt(CRB) {t['sqrt_mean_crb']:.4f} ratio {t['ratio_rmse_over_bound']:.2f} std(z) {t['std_z']:.2f}")
    print(json.dumps(summary["overall"], indent=1))


if __name__ == "__main__":
    main()
