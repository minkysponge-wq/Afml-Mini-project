"""Figures + summary for Phase A from saved results (no re-optimisation).

Usage: python -m blinit.experiments.phase_a_report [--dir results/phaseA]
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

from blinit.observe import ObsConfig
from blinit.simulator import SimConfig, initial_condition, interp_periodic, solve

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4 (fixed order)
DIAG = "#8a8985"                                        # neutral for the diagnostic oracle
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3dd"

plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "lines.linewidth": 1.6, "legend.frameon": False})


def load_runs(d):
    with open(d / "runs.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def summarize(rows):
    out = {}
    for m in dict.fromkeys(r["method"] for r in rows):
        rs = [r for r in rows if r["method"] == m]
        g = lambda k: np.array([r[k] for r in rs])
        out[m] = dict(n=len(rs),
                      rel_err_median=float(np.median(g("rel_err"))), rel_err_p90=float(np.percentile(g("rel_err"), 90)),
                      rel_err_max=float(g("rel_err").max()),
                      solves_median=float(np.median(g("n_solves"))), solves_p90=float(np.percentile(g("n_solves"), 90)),
                      wall_median_s=float(np.median(g("wall_seconds"))), wall_p90_s=float(np.percentile(g("wall_seconds"), 90)),
                      sim_failures=int(g("sim_failures").sum()),
                      not_success=int(sum(1 for r in rs if str(r.get("success", "True")) in ("False", "0.0"))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/phaseA")
    d = Path(ap.parse_args().dir)
    fig_dir = d / "figures"
    fig_dir.mkdir(exist_ok=True)
    rows = load_runs(d)
    P = np.load(d / "problems.npz")
    meta = json.loads((d / "meta.json").read_text())
    methods = list(dict.fromkeys(r["method"] for r in rows))
    color = {m: (DIAG if "ORACLE" in m else SERIES[i]) for i, m in enumerate(methods)}

    summ = summarize(rows)
    (d / "summary.json").write_text(json.dumps(summ, indent=2))

    # 1. representative trajectories + sensors (low / mid / high nu)
    sim, obs = SimConfig.from_json(), ObsConfig.from_json()
    order = np.argsort(P["nu"])
    picks = [order[0], order[len(order) // 2], order[-1]]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    tcol = ["#9ec5f4", "#5598e7", "#256abf", "#104281"]
    for ax, i in zip(axs, picks):
        u0 = initial_condition(sim.xc, P["amps"][i], P["phases"][i])
        traj = solve(u0, P["nu"][i], obs.times, sim)
        ax.plot(sim.xc, u0, color=MUTED, lw=1, ls="--", label="t=0")
        for j, t in enumerate(obs.times):
            ax.plot(sim.xc, traj[j], color=tcol[j], label=f"t={t}")
            ax.plot(P["sensor_x"], P["y_obs"][i, j], "o", ms=3.5, color=tcol[j], mec="white", mew=0.6)
        ax.set_title(f"problem {i}, nu={P['nu'][i]:.4f}", color=INK)
        ax.set_xlabel("x")
    axs[0].set_ylabel("u"); axs[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Trajectories at observation times; dots = 16 sensor readings", color=INK)
    fig.tight_layout(); fig.savefig(fig_dir / "trajectories.png", dpi=150); plt.close(fig)

    # 2. objective vs viscosity
    fig, ax = plt.subplots(figsize=(6, 4))
    nu_scan = np.exp(P["scan_theta"])
    for i in range(len(P["nu"])):
        J = P["scan_objective"][i]
        ax.plot(nu_scan, J / J.max(), color=SERIES[0], alpha=0.45, lw=1)
        ax.axvline(P["nu"][i], color=GRID, lw=0.6, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("candidate nu (PDEBench convention)"); ax.set_ylabel("objective / max over scan")
    ax.set_title(f"Observation mismatch vs nu ({len(P['nu'])} problems, 16 sensors, noise-free)", color=INK)
    fig.tight_layout(); fig.savefig(fig_dir / "objective_curves.png", dpi=150); plt.close(fig)

    # 3. true vs estimated + 4. cost
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        axs[0].plot([r["nu_true"] for r in rs], [r["nu_hat"] for r in rs], "o", ms=5, mfc="none",
                    color=color[m], label=m)
        axs[1].plot([r["nu_true"] for r in rs], [max(r["rel_err"], 1e-16) for r in rs], "o", ms=5,
                    color=color[m], label=m)
    axs[0].plot([0.01, 0.1], [0.01, 0.1], color=MUTED, lw=1)
    for a in axs[:2]:
        a.set_xscale("log"); a.set_xlabel("true nu")
    axs[0].set_yscale("log"); axs[0].set_ylabel("estimated nu"); axs[0].legend(fontsize=7)
    axs[1].set_yscale("log"); axs[1].set_ylabel("relative error")
    pos = np.arange(len(methods))
    for k, m in enumerate(methods):
        s = [r["n_solves"] for r in rows if r["method"] == m]
        axs[2].boxplot([s], positions=[k], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=color[m], alpha=0.35, color=color[m]),
                       medianprops=dict(color=INK), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                       flierprops=dict(markeredgecolor=MUTED))
    axs[2].set_xticks(pos, [m.replace("(diagnostic)", "\n(diag.)") for m in methods], fontsize=7)
    axs[2].set_ylabel("forward solves per problem")
    fig.tight_layout(); fig.savefig(fig_dir / "recovery_and_cost.png", dpi=150); plt.close(fig)

    # 5. grid convergence
    gc = meta.get("grid_convergence")
    if gc:
        fig, ax = plt.subplots(figsize=(6, 4))
        nus = sorted({r["nu"] for r in gc})
        for k, nu in enumerate(nus):
            for ic in sorted({r["ic"] for r in gc}):
                rs = [r for r in gc if r["nu"] == nu and r["ic"] == ic]
                ax.plot([r["nx"] for r in rs], [r["rms_err"] for r in rs], "-o", ms=4, color=SERIES[k],
                        label=f"nu={nu}" if ic == 0 else None)
                ax.axhline(rs[0]["rms_sens_1pct"], color=SERIES[k], ls=":", lw=1)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("nx"); ax.set_ylabel("RMS sensor error vs nx=1024")
        ax.set_title("Grid convergence at sensors (dotted: obs change for +1% nu)", color=INK)
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(fig_dir / "grid_convergence.png", dpi=150); plt.close(fig)

    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
