"""Build the presentation video: demo, datasets, benchmarks and results.

Usage:
  python docs/video/demo_run.py --variant 4 --json docs/video/demo_trace.json > docs/video/demo_output.txt
  python -m pytest -v | (filter) > docs/video/pytest_output.txt      (already captured; see README in this folder)
  python docs/video/build_video.py            # frames -> docs/video/frames/, deck -> docs/video/AFML_video.pptx
  powershell -ExecutionPolicy Bypass -File docs/video/export_video.ps1   # -> docs/video/AFML_project_video.mp4

Every number on a frame comes from a committed result file (named in the frame's source line) or from the
paper (docs/report/sec_results.tex), which names its own sources. The demo frames show real program output.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FR = HERE / "frames"
FIGP = ROOT / "docs" / "report" / "figures"
FIGR = ROOT / "results" / "report" / "figures"

W, H, DPI = 19.2, 10.8, 100                       # 1920 x 1080 px
NAVY, RED, INK, BODY, MUTED = "#1F3A93", "#E8141B", "#1C2433", "#3A4252", "#6B7280"
TEAL, PAPER, BAND = "#2BB5C0", "#FFFFFF", "#F2F4F8"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
TERM_BG, TERM_FG, TERM_DIM, TERM_OK = "#1E1E1E", "#E6E6E6", "#9DA5B4", "#7EE787"
SANS, MONO = "Segoe UI", "Consolas"
LOGO = mpimg.imread(HERE / "pes_logo.png")
US = 144.3e-6

frames = []          # (png path, seconds)


def canvas(bg=PAPER):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=bg)
    return fig


def logo(fig):
    ax = fig.add_axes([0.905, 0.835, 0.075, 0.145]); ax.imshow(LOGO); ax.axis("off")


def footer(fig, n, source=None, dark=False):
    c = TERM_DIM if dark else MUTED
    fig.text(0.5, 0.025, "UE24AM342AA1 Course Project", ha="center", fontsize=15, color=c, family=SANS)
    fig.text(0.965, 0.025, str(n), ha="right", fontsize=15, color=c, family=SANS)
    if source:
        fig.text(0.035, 0.025, "Source: " + source, ha="left", fontsize=13, color=c, family=SANS)


def heading(fig, title, kicker=None):
    if kicker:
        fig.text(0.05, 0.915, kicker.upper(), fontsize=20, color=TEAL, family=SANS, weight="bold")
    fig.text(0.05, 0.85, title, fontsize=40, color=INK, family=SANS, weight="bold", va="center")
    fig.add_artist(plt.Line2D([0.05, 0.86], [0.79, 0.79], color=TEAL, lw=4))


def caption(fig, text):
    fig.patches.append(FancyBboxPatch((0.05, 0.065), 0.90, 0.075, boxstyle="round,pad=0.004,rounding_size=0.01",
                                      transform=fig.transFigure, facecolor=BAND, edgecolor="none"))
    fig.text(0.07, 0.1025, text, fontsize=22, color=INK, family=SANS, va="center", wrap=True)


def bullets(fig, items, x=0.06, y=0.72, size=26, gap=0.085, width=70, color=BODY):
    for it in items:
        lines = textwrap.wrap(it, width)
        fig.text(x, y, "•", fontsize=size, color=TEAL, family=SANS, va="top")
        fig.text(x + 0.022, y, "\n".join(lines), fontsize=size, color=color, family=SANS, va="top", linespacing=1.3)
        y -= gap + 0.045 * (len(lines) - 1) * size / 26


def image(fig, path, rect):
    ax = fig.add_axes(rect); ax.imshow(mpimg.imread(path)); ax.axis("off")
    return ax


def table(fig, rows, rect, col_w, size=22, header_color=NAVY, highlight=None):
    ax = fig.add_axes(rect); ax.axis("off")
    n = len(rows); h = 1.0 / n
    for i, row in enumerate(rows):
        y = 1 - (i + 1) * h
        fc = header_color if i == 0 else ("#E3ECFA" if highlight and i in highlight else ("#F6F7FA" if i % 2 else "#FFFFFF"))
        ax.add_patch(Rectangle((0, y), 1, h, facecolor=fc, edgecolor="#D5DAE3", lw=1, transform=ax.transAxes))
        x = 0
        for j, cell in enumerate(row):
            ax.text(x + 0.012, y + h / 2, cell, fontsize=size, va="center", family=SANS,
                    color="#FFFFFF" if i == 0 else INK, weight="bold" if i == 0 else "normal", transform=ax.transAxes)
            x += col_w[j]


def save(fig, name, seconds):
    p = FR / f"{len(frames) + 1:02d}_{name}.png"
    fig.savefig(p, dpi=DPI, facecolor=fig.get_facecolor()); plt.close(fig)
    frames.append((p, seconds))


def section(title, sub, n):
    fig = canvas(NAVY)
    fig.text(0.07, 0.55, title, fontsize=72, color="#FFFFFF", family=SANS, weight="bold")
    fig.text(0.07, 0.44, sub, fontsize=30, color="#C9D4F0", family=SANS)
    footer(fig, n, dark=True)
    save(fig, "section_" + title.lower().replace(" ", "_"), 4)


def terminal(lines, title, cap, n, name, seconds, size=21, cmd=None):
    fig = canvas("#FFFFFF")
    heading(fig, title, "Demo")
    logo(fig)
    ax = fig.add_axes([0.05, 0.16, 0.90, 0.60]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", facecolor=TERM_BG,
                                edgecolor="none", transform=ax.transAxes))
    for k, c in enumerate(("#FF5F56", "#FFBD2E", "#27C93F")):
        ax.add_patch(plt.Circle((0.02 + k * 0.018, 0.955), 0.007, color=c, transform=ax.transAxes))
    y, dy = 0.89, (size * 1.45) / (0.60 * H * DPI)
    if cmd:
        ax.text(0.015, y, "PS Afml-Mini-project> " + cmd, fontsize=size, family=MONO, color=TERM_OK, va="top")
        y -= dy * 1.3
    for ln in lines:
        col = TERM_OK if ("PASS" in ln or "passed" in ln) else (TERM_DIM if ln.startswith(("==", "--")) else TERM_FG)
        ax.text(0.015, y, ln, fontsize=size, family=MONO, color=col, va="top")
        y -= dy
    caption(fig, cap)
    footer(fig, n)
    save(fig, name, seconds)


def main():
    FR.mkdir(parents=True, exist_ok=True)
    for f in FR.glob("*.png"):
        f.unlink()
    demo = json.loads((HERE / "demo_trace.json").read_text())
    demo_txt = (HERE / "demo_output.txt").read_text(encoding="utf-8").splitlines()
    pytest_txt = (HERE / "pytest_output.txt").read_text(encoding="utf-8").splitlines()
    n = 0

    # 1 title
    n += 1; fig = canvas(); logo(fig)
    fig.text(0.05, 0.86, "UE24AM342AA1 – Advanced Foundations in Machine Learning", fontsize=34, color=RED, family=SANS, weight="bold")
    fig.text(0.07, 0.66, "Does a Learned Starting Point Make\nViscosity Estimation Cheaper?", fontsize=56, color=NAVY,
             family=SANS, weight="bold", va="center", linespacing=1.15)
    fig.text(0.07, 0.52, "A cost-counted study on the Burgers equation — demo, datasets, benchmarks and results",
             fontsize=28, color=BODY, family=SANS)
    team = ["PES1UG24AM246  Sampreeth Jaladi", "PES1UG24AM243  Samarth", "PES1UG24AM255  Sarva Sandip",
            "PES1UG24AM239  Sai Likhith S"]
    for k, t in enumerate(team):
        fig.text(0.07, 0.40 - k * 0.06, t, fontsize=28, color=NAVY, family=SANS)
    footer(fig, n); save(fig, "title", 10)

    # 2 outline
    n += 1; fig = canvas(); logo(fig); heading(fig, "What this video covers")
    items = [("1", "The problem", "estimate a fluid's viscosity from a few noisy sensors"),
             ("2", "Datasets", "how we generated 3,000 + 100 + 150 trajectories, and why"),
             ("3", "Live demo", "tests, the dataset, and one test case solved three ways"),
             ("4", "Benchmarks", "five methods plus one follow-up, and how we kept the comparison fair"),
             ("5", "Results", "pre-registered test, tolerance curves, follow-up, robustness")]
    for k, (num, a, b) in enumerate(items):
        y = 0.68 - k * 0.11
        fig.text(0.07, y, num, fontsize=44, color=TEAL, family=SANS, weight="bold", va="center")
        fig.text(0.12, y + 0.015, a, fontsize=32, color=INK, family=SANS, weight="bold", va="center")
        fig.text(0.12, y - 0.032, b, fontsize=24, color=BODY, family=SANS, va="center")
    footer(fig, n); save(fig, "outline", 14)

    # 3 problem
    n += 1; fig = canvas(); logo(fig); heading(fig, "The problem: find the viscosity ν", "Problem")
    fig.text(0.06, 0.725, r"$u_t + (u^2/2)_x = (\nu/\pi)\,u_{xx}$  on a periodic loop,   $\nu \in [0.01,\ 0.1]$  (PDEBench convention)",
             fontsize=28, color=INK)
    image(fig, FIGP / "trajectories.png", [0.05, 0.30, 0.90, 0.39])
    bullets(fig, ["Known: the starting wave (3 sine modes).  Observed: velocity at 16, 8 or 4 sensors at 4 times, 0–5% noise.  Unknown: ν."],
            y=0.27, size=23, width=130)
    caption(fig, "Low ν keeps sharp fronts; high ν smooths the wave fast. Each simulator run costs up to ~1 s, so runs are the cost.")
    footer(fig, n, "results/phaseA/figures/trajectories.png"); save(fig, "problem", 30)

    # 4 approach
    n += 1; fig = canvas(); logo(fig); heading(fig, "Our approach", "Problem")
    image(fig, FIGP / "pipeline.png", [0.05, 0.17, 0.90, 0.60])
    caption(fig, "An MLP guesses log ν in 0.1 ms; least squares finishes the fit. Brent's search needs no guess. Every run is counted.")
    footer(fig, n, "docs/report/figures/pipeline.png"); save(fig, "approach", 30)

    # 5 section datasets
    n += 1; section("Datasets", "Generated by our own solver — no public dataset fits this question", n)

    # 6 dataset table
    n += 1; fig = canvas(); logo(fig); heading(fig, "Main dataset: phaseB_v2", "Datasets")
    rows = [["Split", "Trajectories", "Cases (× 9 variants)", "Seeds"],
            ["Train", "3,000", "27,000", "IC seeds 0 – 2,999"],
            ["Validation", "50", "450", "IC seeds 1,000,000 – 1,000,049"],
            ["Test (pilot)", "50", "450", "IC seeds 2,000,000 – 2,000,049"]]
    table(fig, rows, [0.06, 0.53, 0.80, 0.23], [0.2, 0.2, 0.27, 0.33], size=24)
    bullets(fig, ["Each case: 64 readings (16 sensor slots × 4 times); 4, 8 or 16 sensors on; 0%, 1% or 5% noise; target log ν, ν log-uniform in [0.01, 0.1].",
                  "Split by starting-wave seed, so no trajectory appears in two splits; normaliser fitted on train only.",
                  "9 automated integrity checks pass; the dataset regenerates bit-identically from configs/phaseB_v2.json (~4 min on 8 cores)."],
            y=0.47, size=23, width=120, gap=0.08)
    caption(fig, "Why not PDEBench's released data: its files fix one viscosity each and use different starting states.")
    footer(fig, n, "data/phaseB_v2/integrity.json, configs/phaseB_v2.json"); save(fig, "dataset", 32)

    # 7 dataset sample
    n += 1
    T = dict(np.load(ROOT / "data" / "phaseB_v2" / "traj_test.npz")); O = dict(np.load(ROOT / "data" / "phaseB_v2" / "obs_test.npz"))
    j = 0; xc = (np.arange(256) + 0.5) / 256; xs = (np.arange(16) + 0.5) / 16
    fig = canvas(); logo(fig); heading(fig, "One trajectory, nine observation variants", "Datasets")
    shades = ["#9CC3F0", "#5E9BE3", "#2A78D6", "#174A8C"]
    for col, (ns, lab) in enumerate([(16, "16 sensors"), (8, "8 sensors"), (4, "4 sensors")]):
        ax = fig.add_axes([0.06 + col * 0.305, 0.22, 0.27, 0.50])
        v = [i for i in range(len(O["parent"])) if O["parent"][i] == j and O["n_sensors"][i] == ns and O["noise"][i] == 0.05][0]
        for k in range(4):
            ax.plot(xc, T["u_at_obs_times"][j][k], color=shades[k], lw=2, label=f"t = {T['times'][k]:g}")
            m = ~np.isnan(O["y"][v][k])
            ax.scatter(xs[m], O["y"][v][k][m], color=shades[k], s=60, edgecolor="#FFFFFF", lw=1.2, zorder=3)
        ax.set_title(f"{lab}, 5% noise", fontsize=22, family=SANS, color=INK)
        ax.set_xlabel("x", fontsize=18); ax.tick_params(labelsize=14)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#E4E3DD", lw=0.6)
        if col == 0:
            ax.set_ylabel("velocity u", fontsize=18); ax.legend(fontsize=14, frameon=False, loc="lower left")
    caption(fig, f"Test trajectory 0 (true ν = {T['nu'][j]:.4f}). Lines: simulated flow; dots: the noisy readings an estimator sees.")
    footer(fig, n, "data/phaseB_v2 (test split)"); save(fig, "dataset_sample", 28)

    # 8 robustness datasets
    n += 1; fig = canvas(); logo(fig); heading(fig, "Three robustness datasets (pre-registered)", "Datasets")
    rows = [["Set", "What changes", "Size"],
            ["Higher-frequency starts", "sine modes 2–4 instead of 1–3", "50 trajectories"],
            ["Finer-grid data", "simulated on 1,024 cells, fitted on 256", "50 trajectories"],
            ["Unseen sensor patterns", "4 new layouts (+2 seen ones as a control)", "50 trajectories"]]
    table(fig, rows, [0.06, 0.42, 0.86, 0.30], [0.30, 0.50, 0.20], size=24)
    caption(fig, "New starting-state seeds; models frozen; analysis rules committed before these sets were run. Fine grid = model mismatch.")
    footer(fig, n, "configs/phaseF_*.json, docs/preregistration_phaseF.md"); save(fig, "robust_sets", 22)

    # 9 section demo
    n += 1; section("Live demo", "Real output from the repository's code", n)

    # 10 pytest
    n += 1
    terminal(pytest_txt[:11] + ["   ... (13 more PASSED)"] + pytest_txt[-6:], "Test suite: 29 tests pass",
             "Solver checked against the exact Cole–Hopf solution, conservation, symmetry; data-leak and model tests.",
             n, "demo_tests", 18, size=20, cmd="python -m pytest -v")

    # 11 dataset demo
    n += 1
    assert demo_txt[5].startswith("integrity checks: "), demo_txt[5]
    integ = demo_txt[5].replace("integrity checks: ", "").split(", ")
    rp = demo_txt[4].split("; ")
    lines = demo_txt[:4] + [rp[0] + ";", "   " + "; ".join(rp[1:]), "integrity checks:"] + \
        ["   " + ", ".join(integ[i:i + 2]) for i in range(0, len(integ), 2)]
    terminal(lines, "Loading the dataset", "Three splits, 9 variants per trajectory, all 9 integrity checks pass.",
             n, "demo_dataset", 18, size=22, cmd="python docs/video/demo_run.py --variant 4")

    # 12-15 live solve animation
    scan = np.array(demo["scan"]); nus = np.exp(scan[:, 0]); J = scan[:, 1]
    meths = [("Brent (full range)", BLUE), ("MLP start + least squares", GREEN), ("MLP bracket + Brent", ORANGE)]
    maxk = max(len(demo["methods"][m]["trace"]) for m, _ in meths)
    nu_t, nu_m = demo["nu_true"], np.exp(demo["theta_mlp"])
    n += 1
    for k in range(0, maxk + 1):
        fig = canvas(); logo(fig)
        heading(fig, f"One test case, solved three ways ({demo['n_sensors']} sensors, {demo['noise']:.0%} noise)", "Demo")
        for c, (m, colr) in enumerate(meths):
            tr = demo["methods"][m]["trace"]; kk = min(k, len(tr))
            ax = fig.add_axes([0.06 + c * 0.31, 0.30, 0.27, 0.41])
            ax.plot(nus, J, color="#B8BEC8", lw=2, zorder=1)
            ax.axvline(nu_t, color=INK, lw=1.5, ls="--", zorder=1)
            if "MLP" in m:
                ax.axvline(nu_m, color=GREEN, lw=1.5, ls=":", zorder=1)
            if "bracket" in m:
                ax.axvspan(np.exp(demo["theta_mlp"] - 0.5), np.exp(demo["theta_mlp"] + 0.5), color=ORANGE, alpha=0.10, zorder=0)
            if kk:
                pts = np.array(tr[:kk])
                ax.scatter(np.exp(pts[:, 0]), pts[:, 1], s=90, color=colr, edgecolor="#FFFFFF", lw=1.5, zorder=3)
                ax.scatter(np.exp(pts[-1, 0]), pts[-1, 1], s=260, facecolor="none", edgecolor=colr, lw=2.5, zorder=4)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.0095, 0.105)
            ax.set_xlabel("candidate ν", fontsize=17); ax.tick_params(labelsize=13)
            if c == 0:
                ax.set_ylabel("misfit J", fontsize=17)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(m, fontsize=22, color=colr, family=SANS, weight="bold")
            cost = tr[kk - 1][2] * US if kk else 0.0
            done = kk == len(tr)
            fig.text(0.06 + c * 0.31, 0.185, f"runs {kk:2d} | {cost:4.2f} CPU-s" + ("  (done)" if done else ""),
                     fontsize=22, family=MONO, color=INK)
        cap = ("Grey: misfit curve. Dashed: true ν = %.4f. Dotted: MLP guess (%.1f%% off). Shaded: the ±0.5 bracket."
               % (nu_t, 100 * abs(nu_m - nu_t) / nu_t))
        if k == maxk:
            ms = [demo["methods"][m] for m, _ in meths]
            cap = ("Same estimate (ν = %.5f) for all three. Brent %d runs / %.2f CPU-s, MLP+LSQ %d / %.2f, bracket %d / %.2f. One case only."
                   % (ms[0]["nu_hat"], ms[0]["n_solves"], ms[0]["cpu_s"], ms[1]["n_solves"], ms[1]["cpu_s"],
                      ms[2]["n_solves"], ms[2]["cpu_s"]))
        caption(fig, cap); footer(fig, n, "docs/video/demo_trace.json (matches results/*_test/runs.csv, case 4)")
        save(fig, f"demo_solve_{k:02d}", 2.2 if k < maxk else 16)

    # 16 solve log
    n += 1
    blk = [l for l in demo_txt[6:] if l.strip()]
    keep = [l for l in blk if l.startswith(("==", "MLP guess", "--", "  =>"))]
    terminal(keep, "The same run, as printed",
             "Case chosen by a fixed rule (first test trajectory, 8 sensors, 1% noise), not hand-picked. Averages follow.",
             n, "demo_log", 18, size=22, cmd="python docs/video/demo_run.py --variant 4")

    # 17 section benchmarks
    n += 1; section("Benchmarks", "Five methods plus one follow-up; one simulator, one cost counter", n)

    # 18 methods
    n += 1; fig = canvas(); logo(fig); heading(fig, "The methods we compare", "Benchmarks")
    rows = [["Method", "Starting point", "Finisher"],
            ["Brent (strongest baseline)", "none: searches the whole range", "bounded Brent search, xatol 1e-3"],
            ["Midpoint + LSQ", "centre of the range", "least squares (trust region)"],
            ["Grid + LSQ", "best of 5 log-spaced runs", "least squares"],
            ["kNN + LSQ", "10 nearest training cases", "least squares"],
            ["MLP + LSQ (3 seeds)", "2×64 MLP prediction", "least squares"],
            ["MLP bracket + Brent (follow-up)", "MLP guess ± 0.5 in log ν", "Brent inside the bracket"]]
    table(fig, rows, [0.06, 0.25, 0.86, 0.50], [0.34, 0.33, 0.33], size=22, highlight={1})
    caption(fig, "LSQ stops at xtol 1e-3 (primary) and 1e-4 (accuracy-matched), bracketing Brent's precision.")
    footer(fig, n, "configs/phaseD_v1_test.json, configs/phaseD_v1_tightlsq_test.json"); save(fig, "methods", 30)

    # 19 fairness
    n += 1; fig = canvas(); logo(fig); heading(fig, "How we kept the comparison fair", "Benchmarks")
    cards = [("One counter", "Every simulator run is counted, including the extra runs least squares uses for its derivative."),
             ("One cost unit", "CPU-s = solver steps × 144.3 µs. Training data, tuning and training are added as offline cost."),
             ("Pre-registered", "Success rules committed to git before each test run; each test set used once per setting."),
             ("Paired statistics", "Per-case cost differences; 95% intervals by resampling whole trajectories (4,000 times).")]
    for k, (a, b) in enumerate(cards):
        x = 0.06 + (k % 2) * 0.445; y = 0.50 - (k // 2) * 0.27
        fig.patches.append(FancyBboxPatch((x, y), 0.415, 0.22, boxstyle="round,pad=0.004,rounding_size=0.015",
                                          transform=fig.transFigure, facecolor=BAND, edgecolor="none"))
        fig.text(x + 0.02, y + 0.17, a, fontsize=30, color=NAVY, family=SANS, weight="bold", va="center")
        fig.text(x + 0.02, y + 0.075, "\n".join(textwrap.wrap(b, 52)), fontsize=21, color=BODY, family=SANS, va="center", linespacing=1.35)
    footer(fig, n, "docs/preregistration_pilot_test.md"); save(fig, "fairness", 30)

    # 20 headroom
    n += 1; fig = canvas(); logo(fig); heading(fig, "Before training: how much can a good start save?", "Benchmarks")
    image(fig, FIGP / "objective_curves.png", [0.05, 0.17, 0.46, 0.59])
    bullets(fig, ["Misfit has one minimum in every problem checked (20 noise-free + 160 noisy).",
                  "All methods reach the same optimum: hard cases lack information, not optimiser skill.",
                  "A start at the true ν still needs ~6 runs under noise vs 9–10 for Brent: at best ~1.5×."],
            x=0.54, y=0.72, size=23, width=44, gap=0.15)
    caption(fig, "This ceiling was measured on development data before any model was trained.")
    footer(fig, n, "results/phaseA, results/phaseA_noise"); save(fig, "headroom", 30)

    # 21 section results
    n += 1; section("Results", "Pilot test: 50 trajectories × 9 conditions = 450 cases", n)

    # 22 test table
    n += 1; fig = canvas(); logo(fig); heading(fig, "Pre-registered test: all methods equally accurate", "Results")
    rows = [["Method", "Error median / p90", "Runs (median)", "CPU-s mean / median"],
            ["Brent", "0.22% / 2.6%", "9", "9.01 / 8.12"],
            ["Midpoint + LSQ", "0.22% / 2.6%", "10", "11.06 / 7.37"],
            ["Grid + LSQ", "0.22% / 2.6%", "12", "12.79 / 11.75"],
            ["kNN + LSQ", "0.22% / 2.6%", "8", "9.27 / 7.65"],
            ["MLP + LSQ (seed 0)", "0.22% / 2.6%", "8", "8.20 / 6.75"]]
    table(fig, rows, [0.06, 0.30, 0.86, 0.44], [0.28, 0.26, 0.20, 0.26], size=24, highlight={5})
    caption(fig, "Primary tolerance. No simulator failures. Since accuracy is identical, the comparison is about cost alone.")
    footer(fig, n, "results/phaseD_v1_3seeds_test/analysis.json"); save(fig, "test_table", 32)

    # 23 paired
    n += 1; fig = canvas(); logo(fig); heading(fig, "Verdict: no reliable saving over Brent", "Results")
    image(fig, FIGR / "test_paired_vs_brent.png", [0.04, 0.155, 0.92, 0.62])
    caption(fig, "Loose stop: 0.80–1.07 CPU-s saved, all seeds. Matched stop: every interval crosses 0. Pre-registered: NOT positive; 2× not met.")
    footer(fig, n, "results/phaseD_v1*_3seeds_test/analysis.json"); save(fig, "paired", 40)

    # 24 numbers
    n += 1; fig = canvas(); logo(fig); heading(fig, "The numbers behind the verdict", "Results")
    stats = [("1.13–1.22×", "median speed-up vs Brent,\nloose stop (target was 2×)"),
             ("1.01–1.05×", "median speed-up vs Brent,\naccuracy-matched stop"),
             ("1.9–2.5 CPU-s", "saved per query vs a midpoint\nstart: the guess helps LSQ")]
    for k, (a, b) in enumerate(stats):
        x = 0.06 + k * 0.30
        fig.text(x, 0.58, a, fontsize=56, color=BLUE if k < 2 else GREEN, family=SANS, weight="bold")
        fig.text(x, 0.54, b, fontsize=24, color=BODY, family=SANS, va="top", linespacing=1.35)
    bullets(fig, ["Break-even vs Brent at the loose stop: 2,790 / 2,625 / 4,455 queries (seeds 0/1/2). At the matched stop: never."],
            y=0.28, size=24, width=110)
    caption(fig, "Offline cost: 2,254 CPU-s for data generation, tuning and training (seed 0).")
    footer(fig, n, "docs/report/sec_results.tex §VI-B"); save(fig, "numbers", 30)

    # 25 tolerance
    n += 1; fig = canvas(); logo(fig); heading(fig, "Why it depends on the stopping rule", "Results")
    image(fig, FIGP / "tolerance_curves.png", [0.04, 0.16, 0.92, 0.61])
    caption(fig, "At equal typical (p90) precision MLP+LSQ is 1.17–1.22× cheaper; at equal worst-case precision 0.95–1.05×.")
    footer(fig, n, "results/phaseF_tolerance (validation)"); save(fig, "tolerance", 35)

    # 26 bracket
    n += 1; fig = canvas(); logo(fig); heading(fig, "Follow-up: give the guess to Brent as a bracket", "Results")
    image(fig, FIGR / "test_followup_bracket.png", [0.04, 0.17, 0.52, 0.59])
    bullets(fig, ["Brent searches only MLP guess ± 0.5 (log ν), with its own stopping rule: precision matched by design.",
                  "Cheaper than Brent for all 3 seeds: median −0.96 to −1.11 CPU-s, mean saving 8–10%.",
                  "Falls back to the full range in 1.6–2.7% of cases. Break-even ~2,400–3,300 queries."],
            x=0.58, y=0.74, size=22, width=44, gap=0.14)
    caption(fig, "Designed after the main test and run on the same test cases (own pre-registration): weaker evidence.")
    footer(fig, n, "results/followup_bracket_w0.5_3seeds_test/analysis.json"); save(fig, "bracket", 32)

    # 27 CRB
    n += 1; fig = canvas(); logo(fig); heading(fig, "Accuracy is already at the theoretical limit", "Results")
    image(fig, FIGP / "crb_test.png", [0.05, 0.16, 0.42, 0.61])
    bullets(fig, ["Cramér–Rao bound: the best error any unbiased estimator can reach with these sensors.",
                  "Error spread relative to the bound: 0.947 for Brent, 0.944 for MLP+LSQ (1 = bound attained).",
                  "So no method can be more accurate here: only cost is left to compare."],
            x=0.52, y=0.72, size=23, width=44, gap=0.16)
    caption(fig, "300 noisy test cases; points on the dashed line attain the bound.")
    footer(fig, n, "results/phaseF_crb/table_test.json"); save(fig, "crb", 28)

    # 28 robustness
    n += 1; fig = canvas(); logo(fig); heading(fig, "Robustness: the saving survives only one shift", "Results")
    image(fig, FIGP / "robustness.png", [0.03, 0.30, 0.94, 0.47])
    bullets(fig, ["Higher-frequency starts: MLP error triples (28–33%); saving gone.   Finer-grid data: saving holds (−1.19 to −1.41 CPU-s).",
                  "Unseen sensor patterns: MLP alone 49–61% error, no better than the midpoint; saving gone."],
            y=0.27, size=21, width=140, gap=0.06)
    caption(fig, "Refinement always recovers full accuracy; what is lost on unfamiliar inputs is the speed-up.")
    footer(fig, n, "results/phaseD_v1_3seeds_test_phaseF_*/analysis.json"); save(fig, "robustness", 40)

    # 29 conclusion
    n += 1; fig = canvas(NAVY)
    fig.text(0.07, 0.80, "CONCLUSION", fontsize=22, color="#8FB8EE", family=SANS, weight="bold")
    concl = ["A learned guess makes least squares ~2 CPU-s cheaper per query than starting in the middle.",
             "It does not reliably beat Brent's search: pre-registered test not positive; 2× not met.",
             "As a bracket for Brent it saves 8–10% at matched precision (post hoc; needs fresh data).",
             "Both methods reach the Cramér–Rao bound; on a one-valley scalar problem little cost is left to save."]
    y = 0.70
    for c in concl:
        lines = textwrap.wrap(c, 80)
        fig.text(0.07, y, "•", fontsize=30, color=TEAL, family=SANS, va="top")
        fig.text(0.095, y, "\n".join(lines), fontsize=30, color="#F4F6FA", family=SANS, va="top", linespacing=1.3)
        y -= 0.05 + 0.052 * len(lines)
    footer(fig, n, dark=True); save(fig, "conclusion", 32)

    # 30 end
    n += 1; fig = canvas(); logo(fig)
    fig.text(0.5, 0.60, "Thank you", fontsize=80, color=NAVY, family=SANS, weight="bold", ha="center")
    fig.text(0.5, 0.47, "Code, data configs, pre-registrations, results and paper:", fontsize=28, color=BODY, family=SANS, ha="center")
    fig.text(0.5, 0.40, "github.com/minkysponge-wq/Afml-Mini-project", fontsize=32, color=BLUE, family=MONO, ha="center")
    fig.text(0.5, 0.26, "Sampreeth Jaladi · Samarth · Sarva Sandip · Sai Likhith S", fontsize=24, color=MUTED, family=SANS, ha="center")
    footer(fig, n); save(fig, "end", 10)

    # deck
    from pptx import Presentation
    from pptx.util import Emu
    prs = Presentation(); prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)
    for p, _ in frames:
        s = prs.slides.add_slide(prs.slide_layouts[6])
        s.shapes.add_picture(str(p), 0, 0, prs.slide_width, prs.slide_height)
    prs.save(HERE / "AFML_video.pptx")
    (HERE / "timings.json").write_text(json.dumps([s for _, s in frames]))
    tot = sum(s for _, s in frames)
    print(f"{len(frames)} frames, {tot:.0f} s ({int(tot // 60)}:{int(tot % 60):02d})")


if __name__ == "__main__":
    main()
