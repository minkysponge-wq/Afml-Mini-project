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

    # timing contamination: per-step cost far above the typical value (e.g. system sleep)
    us = np.array([1e6 * r["solver_seconds"] / max(r["solver_steps"], 1) for r in rows])
    med_us = float(np.median(us))
    for r, u in zip(rows, us):
        r["timing_flag"] = bool(u > 3 * med_us)
    clean = [r for r in rows if not r["timing_flag"]]
    summ = summarize(rows)
    for m, v in summarize(clean).items():
        summ[m]["wall_median_s_unflagged"] = v["wall_median_s"]
        summ[m]["wall_p90_s_unflagged"] = v["wall_p90_s"]
    summ["_timing"] = dict(median_us_per_step=med_us, n_flagged=int(sum(r["timing_flag"] for r in rows)),
                           flag_rule="us/step > 3x median over all runs")
    # unimodality of the diagnostic scan (number of interior local minima)
    J = P["scan_objective"]
    n_min = [int(np.sum((j[1:-1] < j[:-2]) & (j[1:-1] < j[2:])) + (j[0] < j[1]) + (j[-1] < j[-2])) for j in J]
    summ["_scan"] = dict(local_minima_per_problem=n_min, all_unimodal=all(k == 1 for k in n_min))
    # accuracy-vs-cost: best-so-far (by objective, no ground truth) after k solves, scored vs truth
    taus = [1e-2, 1e-3, 1e-4]
    anytime = {}
    if "trace" in rows[0]:
        for m in methods:
            curves, reach = [], {t: [] for t in taus}
            for r in [r for r in rows if r["method"] == m]:
                tr = json.loads(r["trace"])
                th, best, errs = None, np.inf, []
                for t_, J_, _ in tr:
                    if J_ < best:
                        best, th = J_, t_
                    errs.append(abs(np.exp(th) - r["nu_true"]) / r["nu_true"])
                curves.append(errs)
                for t in taus:
                    k = next((i + 1 for i, e in enumerate(errs) if e <= t), None)
                    reach[t].append(k)
            anytime[m] = curves
            summ[m]["solves_to_reach"] = {f"{t:g}": dict(
                median=float(np.median([k for k in reach[t] if k])) if any(reach[t]) else None,
                n_never=sum(k is None for k in reach[t])) for t in taus}
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

    # 4b. accuracy-vs-cost (anytime) curves
    if anytime:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for m in methods:
            K = max(len(c) for c in anytime[m])
            M = np.full((len(anytime[m]), K), np.nan)
            for i, c in enumerate(anytime[m]):
                M[i, :len(c)] = c
                M[i, len(c):] = c[-1]          # method has stopped: error stays at final value
            k = np.arange(1, K + 1)
            ax.plot(k, np.median(M, 0), color=color[m], label=m, ls="--" if "ORACLE" in m else "-")
            ax.fill_between(k, np.percentile(M, 10, 0), np.percentile(M, 90, 0), color=color[m], alpha=0.12, lw=0)
        ax.set_yscale("log"); ax.set_xlabel("forward solves used"); ax.set_ylabel("relative error of best-so-far nu")
        ax.set_title("Accuracy vs cost (median, 10-90% band)", color=INK); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(fig_dir / "accuracy_vs_solves.png", dpi=150); plt.close(fig)

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
