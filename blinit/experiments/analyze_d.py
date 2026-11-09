"""Phase D/E analysis: accuracy, cost, paired comparisons with trajectory-level bootstrap, break-even.

Usage: python -m blinit.experiments.analyze_d --dir results/phaseD_v1_val
Cost measures: forward solves; solver steps (deterministic); CPU-seconds = steps x us_per_step
(us_per_step measured in sequential Phase A runs on this machine, stated in output).
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
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def load(d):
    rows = list(csv.DictReader(open(d / "runs.csv")))
    for r in rows:
        for k in ("n_solves", "solver_steps", "rel_err", "nu_true", "noise", "n_sensors", "parent",
                  "variant", "sim_failures", "objective"):
            r[k] = float(r[k])
    return rows


def boot_ci(diff_by_parent, n_boot=4000, seed=0):
    """Median over variants of paired differences; CI by resampling parent trajectories."""
    rng = np.random.default_rng(seed)
    parents = list(diff_by_parent)
    est = float(np.median(np.concatenate([diff_by_parent[p] for p in parents])))
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(len(parents), len(parents), replace=True)
        bs.append(np.median(np.concatenate([diff_by_parent[parents[i]] for i in pick])))
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "phaseD_v1_val"))
    ap.add_argument("--us-per-step", type=float, default=144.3, help="sequential Phase A median (unflagged)")
    args = ap.parse_args()
    d = Path(args.dir)
    rows = load(d)
    meta = json.loads((d / "meta.json").read_text())
    methods = list(dict.fromkeys(r["method"] for r in rows))
    by = {(r["method"], int(r["variant"])): r for r in rows}
    variants = sorted({int(r["variant"]) for r in rows})
    # reference optimum per variant: best objective across all methods (no ground truth)
    jref = {v: min(by[(m, v)]["objective"] for m in methods) for v in variants}
    out = dict(us_per_step=args.us_per_step, n_variants=len(variants), methods={})
    for m in methods:
        rs = [by[(m, v)] for v in variants]
        e = np.array([r["rel_err"] for r in rs]); s = np.array([r["n_solves"] for r in rs])
        st = np.array([r["solver_steps"] for r in rs])
        gap = np.array([(r["objective"] - jref[int(r["variant"])]) / max(jref[int(r["variant"])], 1e-30) for r in rs])
        out["methods"][m] = dict(
            rel_err_median=float(np.median(e)), rel_err_p90=float(np.percentile(e, 90)),
            solves_median=float(np.median(s)), solves_p90=float(np.percentile(s, 90)), solves_mean=float(s.mean()),
            cpu_s_median=float(np.median(st) * args.us_per_step * 1e-6),
            cpu_s_p90=float(np.percentile(st, 90) * args.us_per_step * 1e-6),
            cpu_s_mean=float(st.mean() * args.us_per_step * 1e-6),
            sim_failures=int(sum(r["sim_failures"] for r in rs)),
            n_rel_obj_gap_gt_1em3=int(np.sum(gap > 1e-3)))
    # paired comparisons vs each classical baseline, by condition too
    classical = [m for m in ("brent", "midpoint+lsq") if m in methods]
    learned = [m for m in methods if m.startswith(("mlp", "knn"))]
    out["paired"] = {}
    for L in learned:
        for B in classical:
            for key, sel in [("all", lambda r: True)] + [
                    (f"s{s}_n{n:g}", (lambda s, n: (lambda r: r["n_sensors"] == s and r["noise"] == n))(s, n))
                    for s in (4, 8, 16) for n in (0.0, 0.01, 0.05)]:
                dpar = {}
                for v in variants:
                    rl, rb = by[(L, v)], by[(B, v)]
                    if sel(rl):
                        dpar.setdefault(int(rl["parent"]), []).append(
                            (rl["solver_steps"] - rb["solver_steps"]) * args.us_per_step * 1e-6)
                if dpar:
                    dpar = {p: np.array(x) for p, x in dpar.items()}
                    est, lo, hi = boot_ci(dpar)
                    mean_diff = float(np.mean(np.concatenate(list(dpar.values()))))
                    out["paired"][f"{L} - {B} | {key}"] = dict(median_cpu_s=est, ci95=[lo, hi], mean_cpu_s=mean_diff)
    # break-even vs strongest classical baseline (lowest mean CPU-s)
    best_c = min(classical, key=lambda m: out["methods"][m]["cpu_s_mean"])
    ov = meta["init_overhead"]
    integ = json.loads((ROOT / "data" / meta["config"]["dataset"] / "integrity.json").read_text())
    gen_cpu = integ["generation_wall_seconds"]["train"] * integ["workers"]   # upper bound: wall x workers
    tuning = json.loads((ROOT / "models" / meta["config"]["model_version"] / "tuning.json").read_text())
    tune_s = sum(t["train_seconds"] for t in tuning["trials"])
    out["break_even"] = {}
    for L in learned:
        init_name = L.replace("+lsq", "")
        o = ov.get(init_name, {})
        offline = gen_cpu + (tune_s + o.get("train_s", 0.0) if init_name.startswith("mlp") else o.get("build_s", 0.0))
        per_query_init = o.get("per_query_single_s", o.get("per_query_s", 0.0))
        saving = out["methods"][best_c]["cpu_s_mean"] - out["methods"][L]["cpu_s_mean"] - per_query_init
        out["break_even"][L] = dict(
            vs=best_c, offline_cpu_s=offline, offline_note="train-data generation (wall x workers, upper bound) + tuning + training / library build",
            per_query_saving_cpu_s=saving,
            break_even_queries=(offline / saving) if saving > 0 else None,
            verdict="no finite break-even (saving <= 0)" if saving <= 0 else "finite")
    (d / "analysis.json").write_text(json.dumps(out, indent=2))

    # figure: cost and error per method by condition
    conds = [(s, n) for s in (4, 8, 16) for n in (0.0, 0.01, 0.05)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    w = 0.8 / len(methods)
    for k, m in enumerate(methods):
        cpu, err = [], []
        for s, n in conds:
            rs = [by[(m, v)] for v in variants if by[(m, v)]["n_sensors"] == s and by[(m, v)]["noise"] == n]
            cpu.append(np.median([r["solver_steps"] for r in rs]) * args.us_per_step * 1e-6)
            err.append(np.median([r["rel_err"] for r in rs]))
        x = np.arange(len(conds)) + (k - len(methods) / 2 + 0.5) * w
        axs[0].bar(x, cpu, w * 0.9, color=SERIES[k % 6], label=m)
        axs[1].plot(np.arange(len(conds)), err, "o-", color=SERIES[k % 6], label=m, ms=4)
    for a in axs:
        a.set_xticks(np.arange(len(conds)), [f"{s}s\n{n:.0%}" for s, n in conds], fontsize=7)
        a.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("median CPU-s per query (steps x us/step)"); axs[0].legend(fontsize=7, frameon=False)
    axs[1].set_yscale("log"); axs[1].set_ylabel("median relative error vs true nu")
    fig.suptitle(f"Phase D ({meta['split']} split): cost and accuracy by sensors / noise")
    fig.tight_layout(); fig.savefig(d / "cost_accuracy_by_condition.png", dpi=150); plt.close(fig)

    for m, v in out["methods"].items():
        print(f"{m:14s} err med {v['rel_err_median']:.2e} p90 {v['rel_err_p90']:.2e} | solves med {v['solves_median']:4.1f} "
              f"mean {v['solves_mean']:5.2f} | CPU-s mean {v['cpu_s_mean']:.2f} med {v['cpu_s_median']:.2f} p90 {v['cpu_s_p90']:.2f} "
              f"| fails {v['sim_failures']} gaps {v['n_rel_obj_gap_gt_1em3']}")
    for k, v in out["paired"].items():
        if k.endswith("| all"):
            print(f"{k:40s} median dCPU-s {v['median_cpu_s']:+.3f} CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] mean {v['mean_cpu_s']:+.3f}")
    print(json.dumps(out["break_even"], indent=1))


if __name__ == "__main__":
    main()
