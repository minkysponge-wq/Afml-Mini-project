"""Figures made only for the report and slides, from committed result files (no simulator calls).

Usage: python docs/report/make_figures.py
Outputs (docs/report/figures/): pipeline.png, test_paired.png, robustness.png
Other report figures are copied unchanged from results/ (see FIGURES.md).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#555555"


def box(ax, x, y, w, h, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc="white", ec=color, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#111111")


def arrow(ax, p, q, color=GREY, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.3, ls=ls))


def pipeline():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    box(ax, 0.1, 2.9, 2.0, 1.3, "Sparse, noisy readings\n4 / 8 / 16 sensors\nx 4 times\n+ known starting state", GREY, fs=8.5)
    box(ax, 2.9, 3.3, 2.1, 0.9, "MLP (2 x 64)\npredicts log viscosity", GREEN)
    box(ax, 2.9, 2.15, 2.1, 0.8, "midpoint / grid / kNN\nstarting guess", ORANGE)
    box(ax, 5.8, 2.7, 2.0, 1.2, "Least-squares fit\n(trust region, bounded)\ncalls the simulator", ORANGE)
    box(ax, 2.9, 0.75, 4.9, 0.85, "Brent bounded scalar search\n(no starting guess; calls the simulator)", BLUE)
    box(ax, 8.5, 1.7, 1.4, 1.3, "Estimated\nviscosity\n+ cost", GREY)
    box(ax, 5.8, 4.05, 2.0, 0.45, "Burgers simulator (256 cells)", GREY, fs=8)
    arrow(ax, (2.1, 3.7), (2.9, 3.75)); arrow(ax, (2.1, 3.3), (2.9, 2.6)); arrow(ax, (2.1, 3.0), (2.9, 1.3))
    arrow(ax, (5.0, 3.75), (5.8, 3.45), GREEN); arrow(ax, (5.0, 2.55), (5.8, 3.1), ORANGE)
    arrow(ax, (7.8, 3.3), (8.5, 2.6)); arrow(ax, (7.8, 1.2), (8.5, 2.0))
    arrow(ax, (6.8, 4.05), (6.8, 3.9), GREY)
    ax.text(5.0, 0.3, "Every simulator call is counted. Cost = solver steps x 144.3 microseconds. "
            "Offline cost of the MLP (data, tuning, training) is divided by the saving per query to give a break-even count.",
            ha="center", fontsize=7.5, color=GREY, wrap=True)
    fig.tight_layout(); fig.savefig(OUT / "pipeline.png", dpi=200); plt.close(fig)


def test_paired():
    runs = [("primary: LSQ xtol 1e-3", "phaseD_v1_3seeds_test"), ("accuracy-matched: LSQ xtol 1e-4", "phaseD_v1_tightlsq_3seeds_test")]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y, labels = 0, []
    for name, folder in runs:
        a = json.loads((ROOT / "results" / folder / "analysis.json").read_text())
        for s in range(3):
            v = a["paired"][f"mlp_s{s}+lsq - brent | all"]
            ax.errorbar(v["median_cpu_s"], y, xerr=[[v["median_cpu_s"] - v["ci95"][0]], [v["ci95"][1] - v["median_cpu_s"]]],
                        fmt="o", color=GREEN if "primary" in name else ORANGE, capsize=3, ms=5)
            labels.append(f"{name}, seed {s}"); y += 1
    ax.axvline(0, color=GREY, lw=0.8, ls="--")
    ax.set_yticks(range(len(labels)), labels, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("median paired difference, MLP+LSQ minus Brent (CPU-s per query); negative = MLP cheaper", fontsize=7.5)
    ax.set_title("Pilot test, 450 variants: saving vs Brent with 95% bootstrap CI", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "test_paired.png", dpi=200); plt.close(fig)


def robustness():
    sets = [("pilot test", "phaseD_v1_3seeds_test"), ("high-frequency IC", "phaseD_v1_3seeds_test_phaseF_highfreq"),
            ("fine-grid data", "phaseD_v1_3seeds_test_phaseF_finegrid"), ("unseen masks", "phaseD_v1_3seeds_test_phaseF_unseenmask")]
    have = [(n, f) for n, f in sets if (ROOT / "results" / f / "analysis.json").exists()]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    for k, (name, folder) in enumerate(have):
        a = json.loads((ROOT / "results" / folder / "analysis.json").read_text())
        for s in range(3):
            v = a["paired"][f"mlp_s{s}+lsq - brent | all"]
            axs[0].errorbar(k + (s - 1) * 0.18, v["median_cpu_s"],
                            yerr=[[v["median_cpu_s"] - v["ci95"][0]], [v["ci95"][1] - v["median_cpu_s"]]],
                            fmt="o", color=GREEN, capsize=3, ms=4)
        m = a["methods"]
        axs[1].bar(k - 0.2, m["mlp_s0+lsq"]["init_rel_err_median"], 0.4, color=GREEN, label="MLP alone" if k == 0 else None)
        axs[1].bar(k + 0.2, m["mlp_s0+lsq"]["rel_err_median"], 0.4, color=ORANGE, label="MLP + LSQ" if k == 0 else None)
        axs[2].bar(k - 0.2, m["brent"]["cpu_s_median"], 0.4, color=BLUE, label="Brent" if k == 0 else None)
        axs[2].bar(k + 0.2, m["mlp_s0+lsq"]["cpu_s_median"], 0.4, color=GREEN, label="MLP + LSQ" if k == 0 else None)
    axs[0].axhline(0, color=GREY, lw=0.8, ls="--")
    axs[0].set_ylabel("MLP+LSQ minus Brent, median CPU-s (95% CI)", fontsize=7.5)
    axs[1].set_yscale("log"); axs[1].set_ylabel("median relative error in viscosity", fontsize=7.5)
    axs[2].set_ylabel("median CPU-s per query", fontsize=7.5)
    for ax in axs:
        ax.set_xticks(range(len(have)), [n for n, _ in have], fontsize=7, rotation=12)
        ax.spines[["top", "right"]].set_visible(False)
    axs[1].legend(fontsize=7, frameon=False); axs[2].legend(fontsize=7, frameon=False)
    fig.suptitle("Robustness sets at the primary tolerance (3 MLP seeds in the left panel; seed 0 elsewhere)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "robustness.png", dpi=200); plt.close(fig)


COPIES = {
    "trajectories.png": "results/phaseA/figures/trajectories.png",
    "objective_curves.png": "results/phaseA/figures/objective_curves.png",
    "grid_convergence.png": "results/phaseA/figures/grid_convergence.png",
    "accuracy_vs_solves.png": "results/phaseA/figures/accuracy_vs_solves.png",
    "test_cost_accuracy.png": "results/phaseD_v1_3seeds_test/cost_accuracy_by_condition.png",
    "tolerance_curves.png": "results/phaseF_tolerance/tolerance_curves.png",
    "crb_test.png": "results/phaseF_crb/crb_test.png",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline(); test_paired(); robustness()
    lines = ["# Report figures and their sources", "", "| figure | source |", "|---|---|",
             "| pipeline.png, test_paired.png, robustness.png | `docs/report/make_figures.py` (reads `results/*/analysis.json`) |"]
    for dst, src in COPIES.items():
        if (ROOT / src).exists():
            shutil.copyfile(ROOT / src, OUT / dst)
            lines.append(f"| {dst} | copied from `{src}` |")
        else:
            lines.append(f"| {dst} | MISSING: `{src}` not found |")
    (OUT.parent / "FIGURES.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
