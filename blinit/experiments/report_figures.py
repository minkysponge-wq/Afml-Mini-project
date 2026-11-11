"""Report figures from saved results only (no simulation, no re-optimisation).

Usage: python -m blinit.experiments.report_figures
Reads results/phaseD_v1*_3seeds_test/{runs.csv,analysis.json} (+ the follow-up bracket test run if present)
and writes results/report/figures/*.png and results/report/tables.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "report"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical slots 1-4, fixed order
NEUTRAL = "#8a8985"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e1e0d9"
US = 144.3e-6
SETTINGS = {"primary": ("phaseD_v1_3seeds_test", "LSQ xtol 1e-3 (primary)"),
            "strict": ("phaseD_v1_tightlsq_3seeds_test", "LSQ xtol 1e-4 (accuracy-matched)")}
CONDS = [(s, n) for s in (4, 8, 16) for n in (0.0, 0.01, 0.05)]

plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2, "legend.frameon": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def load_rows(d):
    rows = list(csv.DictReader(open(ROOT / "results" / d / "runs.csv")))
    for r in rows:
        for k in ("solver_steps", "n_solves", "rel_err", "noise", "n_sensors", "variant", "parent"):
            r[k] = float(r[k])
        if r.get("init_rel_err") not in (None, ""):
            r["init_rel_err"] = float(r["init_rel_err"])
    return rows


def fig_cost(an, tables):
    order = ["brent", "midpoint+lsq", "grid5+lsq", "knn+lsq", "mlp_s0+lsq", "mlp_s1+lsq", "mlp_s2+lsq"]
    col = lambda m: SERIES[0] if m.startswith("mlp") else (SERIES[1] if m.startswith("knn") else NEUTRAL)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for ax, (key, (d, label)) in zip(axs, SETTINGS.items()):
        M = an[key]["methods"]
        y = np.arange(len(order))[::-1]
        for yi, m in zip(y, order):
            ax.plot([M[m]["cpu_s_median"], M[m]["cpu_s_mean"]], [yi, yi], color=GRID, lw=2, zorder=1)
            ax.scatter(M[m]["cpu_s_median"], yi, s=46, color=col(m), zorder=3, edgecolor="#fcfcfb", lw=2)
            ax.scatter(M[m]["cpu_s_mean"], yi, s=46, facecolor="#fcfcfb", edgecolor=col(m), lw=2, zorder=3)
            ax.text(max(M[m]["cpu_s_mean"], M[m]["cpu_s_median"]) + 0.25, yi, f"{M[m]['cpu_s_mean']:.1f}",
                    va="center", fontsize=7.5, color=MUTED)
        ax.axvline(M["brent"]["cpu_s_mean"], color=NEUTRAL, lw=1, ls=":")
        ax.set_yticks(y, order); ax.set_title(label, loc="left"); ax.grid(axis="y", visible=False)
        ax.set_xlabel("CPU-s per query (solver steps x 144.3 us)")
        tables[f"cost_{key}"] = {m: dict(mean=M[m]["cpu_s_mean"], median=M[m]["cpu_s_median"],
                                         solves_median=M[m]["solves_median"]) for m in order}
    axs[0].scatter([], [], s=46, color=INK, label="median"); axs[0].scatter([], [], s=46, facecolor="none", edgecolor=INK, lw=2, label="mean (labelled)")
    axs[0].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Pilot test: cost per query (450 variants). Dotted line: Brent mean.", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "test_cost_by_method.png", dpi=170); plt.close(fig)


def fig_forest(an, tables):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, sharex=True)
    labels = ["all"] + [f"s{s}_n{n:g}" for s, n in CONDS]
    ticks = ["all variants"] + [f"{s} sensors, {n:.0%} noise" for s, n in CONDS]
    for ax, (key, (d, label)) in zip(axs, SETTINGS.items()):
        P = an[key]["paired"]
        for k in range(3):
            ys = np.arange(len(labels))[::-1] + (1 - k) * 0.22
            for yi, c in zip(ys, labels):
                p = P[f"mlp_s{k}+lsq - brent | {c}"]
                ax.plot(p["ci95"], [yi, yi], color=SERIES[k], lw=2, solid_capstyle="round")
                ax.scatter(p["median_cpu_s"], yi, s=30, color=SERIES[k], zorder=3, edgecolor="#fcfcfb", lw=1.5,
                           label=f"MLP seed {k}" if c == "all" else None)
        ax.axvline(0, color=INK, lw=1)
        ax.set_title(label, loc="left"); ax.grid(axis="y", visible=False)
        ax.set_xlabel("MLP+LSQ minus Brent: median CPU-s per query, 95% CI\n(left of 0 = MLP cheaper)")
        ax.axhline(len(labels) - 1.5, color=GRID, lw=0.8)
        tables[f"paired_mlp_vs_brent_{key}"] = {c: {f"s{k}": P[f"mlp_s{k}+lsq - brent | {c}"] for k in range(3)} for c in labels}
    axs[0].set_yticks(np.arange(len(labels))[::-1], ticks)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper right", ncol=3, fontsize=8, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("Paired saving vs Brent, by condition (bootstrap over test trajectories)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "figures" / "test_paired_vs_brent.png", dpi=170); plt.close(fig)


def fig_breakeven(an, tables):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    q = np.linspace(0, 8000, 400)
    brent = an["primary"]["methods"]["brent"]["cpu_s_mean"]
    ax.plot(q, brent * q / 3600, color=NEUTRAL, label="Brent (no offline cost)")
    ax.text(q[-1], brent * q[-1] / 3600, "  Brent", color=INK, va="center", fontsize=8)
    for k, (key, name) in enumerate([("primary", "MLP s0, xtol 1e-3"), ("strict", "MLP s0, xtol 1e-4")]):
        be = an[key]["break_even"]["mlp_s0+lsq"]
        per = brent - be["per_query_saving_cpu_s"]
        y = (be["offline_cpu_s"] + per * q) / 3600
        ax.plot(q, y, color=SERIES[k], label=name)
        ax.text(q[-1], y[-1], f"  {name}", color=INK, va="center", fontsize=8)
        if be["break_even_queries"]:
            x = be["break_even_queries"]
            ax.scatter(x, brent * x / 3600, s=50, color=SERIES[k], zorder=3, edgecolor="#fcfcfb", lw=2)
            ax.annotate(f"break-even ~{x:,.0f} queries", (x, brent * x / 3600), xytext=(10, -22),
                        textcoords="offset points", fontsize=8, color=INK)
        tables[f"break_even_{key}"] = {f"mlp_s{s}": an[key]["break_even"][f"mlp_s{s}+lsq"] for s in range(3)}
    ax.set_xlabel("number of queries"); ax.set_ylabel("cumulative CPU-hours (incl. offline)")
    ax.set_title("End-to-end cost incl. data generation, tuning and training", loc="left")
    ax.set_xlim(0, 8000); ax.margins(y=0.05)
    fig.tight_layout(rect=(0, 0, 0.83, 1)); fig.savefig(OUT / "figures" / "test_break_even.png", dpi=170); plt.close(fig)


def fig_init_error(rows, tables):
    """Neural-only (initialiser) error on test, by condition."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(CONDS))
    meth = [("mlp_s0+lsq", "MLP seed 0", SERIES[0]), ("knn+lsq", "kNN (k=10)", SERIES[1]), ("midpoint+lsq", "midpoint", NEUTRAL)]
    tables["init_error_test"] = {}
    for m, name, c in meth:
        med = []
        for s, n in CONDS:
            rs = [r for r in rows if r["method"] == m and r["n_sensors"] == s and r["noise"] == n]
            if m == "midpoint+lsq":
                th0 = 0.5 * (np.log(0.01) + np.log(0.1))
                med.append(np.median([abs(np.exp(th0) - float(r["nu_true"])) / float(r["nu_true"]) for r in rs]))
            else:
                med.append(np.median([r["init_rel_err"] for r in rs]))
        ax.plot(x, np.array(med) * 100, "o-", color=c, ms=6, mec="#fcfcfb", mew=1.5, label=name)
        ax.text(x[-1] + 0.15, med[-1] * 100, name, color=INK, va="center", fontsize=8)
        tables["init_error_test"][name] = dict(zip([f"s{s}_n{n:g}" for s, n in CONDS], map(float, med)))
    ax.set_xticks(x, [f"{s}s\n{n:.0%}" for s, n in CONDS], fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("median |error| of the starting guess (%)")
    ax.set_yticks([5, 10, 20, 50], ["5%", "10%", "20%", "50%"]); ax.minorticks_off(); ax.set_ylim(5, 70)
    ax.set_title("Starting-guess error before refinement (pilot test)", loc="left")
    ax.set_xlim(-0.3, len(CONDS) + 0.9)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "test_init_error.png", dpi=170); plt.close(fig)


def fig_bracket(tables):
    d = ROOT / "results" / "followup_bracket_w0.5_3seeds_test"
    if not (d / "analysis.json").exists():
        return
    a = json.loads((d / "analysis.json").read_text())
    an_p = json.loads((ROOT / "results" / SETTINGS["primary"][0] / "analysis.json").read_text())
    an_s = json.loads((ROOT / "results" / SETTINGS["strict"][0] / "analysis.json").read_text())
    rows = [("Brent (full range)", an_p["methods"]["brent"]["cpu_s_mean"], an_p["methods"]["brent"]["cpu_s_median"], NEUTRAL)]
    for s in range(3):
        m = next(k for k in a["methods"] if k.startswith(f"mlp_s{s}_"))
        rows.append((f"MLP s{s} bracket + Brent", a["methods"][m]["cpu_s_mean"], a["methods"][m]["cpu_s_median"], SERIES[2]))
    for s in range(3):
        m = f"mlp_s{s}+lsq"
        rows.append((f"MLP s{s} + LSQ 1e-4", an_s["methods"][m]["cpu_s_mean"], an_s["methods"][m]["cpu_s_median"], SERIES[0]))
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    y = np.arange(len(rows))[::-1]
    for yi, (name, mean, med, c) in zip(y, rows):
        ax.plot([med, mean], [yi, yi], color=GRID, lw=2)
        ax.scatter(med, yi, s=46, color=c, zorder=3, edgecolor="#fcfcfb", lw=2)
        ax.scatter(mean, yi, s=46, facecolor="#fcfcfb", edgecolor=c, lw=2, zorder=3)
        ax.text(max(mean, med) + 0.2, yi, f"{mean:.2f}", va="center", fontsize=7.5, color=MUTED)
    ax.axvline(rows[0][1], color=NEUTRAL, lw=1, ls=":")
    ax.set_yticks(y, [r[0] for r in rows]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("CPU-s per query (filled: median, open: mean, labelled)")
    ax.set_title("Follow-up (test): learned bracket vs learned LSQ start", loc="left")
    ax.set_xlim(right=max(r[1] for r in rows) + 0.35)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "test_followup_bracket.png", dpi=170); plt.close(fig)
    tables["followup_bracket_test"] = a


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    an = {k: json.loads((ROOT / "results" / d / "analysis.json").read_text()) for k, (d, _) in SETTINGS.items()}
    tables = {}
    fig_cost(an, tables)
    fig_forest(an, tables)
    fig_breakeven(an, tables)
    fig_init_error(load_rows(SETTINGS["primary"][0]), tables)
    fig_bracket(tables)
    (OUT / "tables.json").write_text(json.dumps(tables, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
