"""Phase E follow-up: direct Brent search inside a learned bracket [theta_MLP - w, theta_MLP + w].

Brent keeps the baseline's stopping rule (xatol 1e-3), so precision matches the Brent baseline by
construction; only the starting interval changes. If the estimate hits an interior bracket edge, Brent
re-runs on the full bounds (counted).

Usage:
  python -m blinit.experiments.followup_bracket run --config configs/followup_bracket_val.json --split val
  python -m blinit.experiments.followup_bracket analyze --dir results/followup_bracket_val \
      --brent-dir results/phaseD_v1_val

Brent baseline rows are taken from the committed Phase D run on the same split. Results are
deterministic, and re-running Brent here was verified bit-identical on sampled variants. Cost =
solver steps x 144.3 us/step, as in analyze_d.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from blinit.data import features
from blinit.experiments.analyze_d import boot_ci
from blinit.experiments.train_mlp import load_split
from blinit.inverse import Problem, run_method
from blinit.model import load_model, predict_theta
from blinit.observe import ObsConfig
from blinit.simulator import SimConfig

ROOT = Path(__file__).resolve().parents[2]
US_PER_STEP = 144.3


def run_variant(args):
    v, amps, phases, y, mask, nu_true, inits, widths, brent_kw = args
    sim, obs = SimConfig.from_json(), ObsConfig.from_json()
    y_full = np.where(np.isnan(y), 0.0, y)
    mk = lambda: Problem(amps, phases, y_full, mask, sim, obs)
    rows = []
    for name, th0 in inits.items():
        for w in widths:
            r = run_method(f"{name}_w{w:g}+brent_bracket", mk, init_theta=th0, brent_kw=brent_kw, bracket_width=w)
            r.pop("trace", None)
            r.update(variant=v, nu_true=nu_true, rel_err=abs(r["nu_hat"] - nu_true) / nu_true)
            rows.append(r)
    return rows


def cmd_run(args):
    cfg = json.loads(Path(args.config).read_text())
    dset = ROOT / "data" / cfg["dataset"]
    out = ROOT / "results" / f"{cfg['config_version']}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    norm = dict(np.load(dset / "normaliser.npz")); norm["fitted_on"] = str(norm["fitted_on"])
    T, O = load_split(dset, args.split)
    X, _ = features(O, T, norm)
    inits = {}
    for seed in cfg["mlp_seeds"]:
        m = load_model(ROOT / "models" / cfg["model_version"] / f"seed{seed}" / "model.pkl")
        inits[f"mlp_s{seed}"] = predict_theta(m, X)
    n = len(X) if args.limit is None else args.limit
    tasks = [(v, T["amps"][O["parent"][v]], T["phases"][O["parent"][v]], O["y"][v], O["mask"][v],
              float(T["nu"][O["parent"][v]]), {k: float(a[v]) for k, a in inits.items()},
              cfg["widths"], cfg["brent"]) for v in range(n)]
    per_variant = len(inits) * len(cfg["widths"])
    partial = out / "runs.partial.jsonl"
    rows = [json.loads(l) for l in partial.read_text().splitlines()] if partial.exists() else []
    counts = {}
    for r in rows:
        counts[r["variant"]] = counts.get(r["variant"], 0) + 1
    done = {v for v, c in counts.items() if c == per_variant}
    rows = [r for r in rows if r["variant"] in done]
    tasks = [t for t in tasks if t[0] not in done]
    if done:
        print(f"resuming: {len(done)} done, {len(tasks)} to go", flush=True)
    partial.write_text("".join(json.dumps(r) + "\n" for r in rows))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(partial, "a") as fp:
        for k, rs in enumerate(ex.map(run_variant, tasks, chunksize=2)):
            for r in rs:
                v = r["variant"]
                r.update(n_sensors=int(O["n_sensors"][v]), noise=float(O["noise"][v]), parent=int(O["parent"][v]))
            rows += rs
            fp.write("".join(json.dumps(r) + "\n" for r in rs)); fp.flush()
            if (k + 1) % 25 == 0:
                print(f"{len(done) + k + 1}/{n} variants ({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r["variant"], r["method"]))
    keys = sorted({k for r in rows for k in r})
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    (out / "meta.json").write_text(json.dumps(dict(
        config=cfg, split=args.split, n_variants=n, workers=args.workers,
        timing_note="parallel workers: cost from solver steps only",
        finished=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    print("saved", out)


def _load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("solver_steps", "n_solves", "rel_err", "nu_hat", "noise", "n_sensors", "parent", "variant"):
            r[k] = float(r[k])
    return rows


def cmd_analyze(args):
    d, bd = Path(args.dir), Path(args.brent_dir)
    rows, base = _load(d / "runs.csv"), _load(bd / "runs.csv")
    brent = {int(r["variant"]): r for r in base if r["method"] == "brent"}
    variants = sorted({int(r["variant"]) for r in rows})
    cpu = lambda r: r["solver_steps"] * US_PER_STEP * 1e-6
    b = [brent[v] for v in variants]
    out = dict(us_per_step=US_PER_STEP, brent_dir=str(bd), n_variants=len(variants), methods={}, paired={})
    out["methods"]["brent"] = dict(cpu_s_mean=float(np.mean([cpu(r) for r in b])),
                                   cpu_s_median=float(np.median([cpu(r) for r in b])),
                                   solves_median=float(np.median([r["n_solves"] for r in b])),
                                   rel_err_median=float(np.median([r["rel_err"] for r in b])),
                                   rel_err_p90=float(np.percentile([r["rel_err"] for r in b], 90)))
    conds = [("all", lambda r: True)] + [
        (f"s{s}_n{n:g}", (lambda s, n: (lambda r: r["n_sensors"] == s and r["noise"] == n))(s, n))
        for s in (4, 8, 16) for n in (0.0, 0.01, 0.05)]
    for m in sorted({r["method"] for r in rows}):
        rs = {int(r["variant"]): r for r in rows if r["method"] == m}
        x = [rs[v] for v in variants]
        dev = [abs(rs[v]["nu_hat"] - brent[v]["nu_hat"]) / brent[v]["nu_hat"] for v in variants]
        out["methods"][m] = dict(
            cpu_s_mean=float(np.mean([cpu(r) for r in x])), cpu_s_median=float(np.median([cpu(r) for r in x])),
            solves_median=float(np.median([r["n_solves"] for r in x])), solves_mean=float(np.mean([r["n_solves"] for r in x])),
            rel_err_median=float(np.median([r["rel_err"] for r in x])),
            rel_err_p90=float(np.percentile([r["rel_err"] for r in x], 90)),
            fallback_rate=float(np.mean([str(r["fallback"]) == "True" for r in x])),
            sim_failures=int(sum(float(r["sim_failures"]) for r in x)),
            nu_dev_from_brent_median=float(np.median(dev)), nu_dev_from_brent_max=float(np.max(dev)))
        for key, sel in conds:
            dpar = {}
            for v in variants:
                if sel(rs[v]):
                    dpar.setdefault(int(rs[v]["parent"]), []).append(cpu(rs[v]) - cpu(brent[v]))
            dpar = {p: np.array(z) for p, z in dpar.items()}
            est, lo, hi = boot_ci(dpar)
            out["paired"][f"{m} - brent | {key}"] = dict(
                median_cpu_s=est, ci95=[lo, hi], mean_cpu_s=float(np.mean(np.concatenate(list(dpar.values())))))
    (d / "analysis.json").write_text(json.dumps(out, indent=2))
    print(f"{'method':26s} {'CPU mean':>8s} {'CPU med':>8s} {'solves':>6s} {'fallback':>8s} {'err med':>8s} {'dev max':>8s}")
    for m, v in out["methods"].items():
        print(f"{m:26s} {v['cpu_s_mean']:8.2f} {v['cpu_s_median']:8.2f} {v['solves_median']:6.1f} "
              f"{v.get('fallback_rate', 0):8.1%} {v['rel_err_median']:8.2e} {v.get('nu_dev_from_brent_max', 0):8.1e}")
    for k, v in out["paired"].items():
        if k.endswith("| all"):
            print(f"{k:40s} median {v['median_cpu_s']:+.3f} CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] mean {v['mean_cpu_s']:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--split", choices=["val", "test"], required=True)
    r.add_argument("--workers", type=int, default=11)
    r.add_argument("--limit", type=int, default=None)
    a = sub.add_parser("analyze")
    a.add_argument("--dir", required=True)
    a.add_argument("--brent-dir", required=True)
    args = ap.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_analyze(args)


if __name__ == "__main__":
    main()
