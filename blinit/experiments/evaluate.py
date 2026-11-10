"""Phase D: paired comparison of initialisers + shared refinement, and direct Brent search.

Usage:
  python -m blinit.experiments.evaluate --split val  [--workers 12] [--limit N]   # development
  python -m blinit.experiments.evaluate --split test [--workers 12]              # final pilot report (run once)

All methods see the same observation variant, same simulator, bounds and refinement settings.
Ground truth is used only to score. Parallel workers: solve counts/errors valid; wall times are
reported per method but are NOT clean timings (shared CPU) - see timing subset protocol.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from blinit.data import features
from blinit.experiments.train_mlp import load_split
from blinit.inverse import Problem, run_method
from blinit.knn import KNNInit
from blinit.model import load_model, predict_theta
from blinit.observe import ObsConfig
from blinit.simulator import CONFIG_DIR, SimConfig

ROOT = Path(__file__).resolve().parents[2]


def tune_knn(cfg, Xtr, ytr, Xva, yva):
    """Select k / IC weight by neural-only VALIDATION error (no simulator calls)."""
    trials = []
    for k, w in itertools.product(cfg["knn_grid"]["k"], cfg["knn_grid"]["ic_weight"]):
        kn = KNNInit(k, w).fit(Xtr, ytr)
        trials.append(dict(k=k, ic_weight=w, val_rmse_log=float(np.sqrt(np.mean((kn.predict(Xva) - yva) ** 2)))))
    return min(trials, key=lambda t: t["val_rmse_log"]), trials


def run_variant(args):
    v, amps, phases, y, mask, nu_true, inits, cfg = args
    sim, obs = SimConfig.from_json(), ObsConfig.from_json()
    y_full = np.where(np.isnan(y), 0.0, y)      # values outside the mask are never read
    mk = lambda: Problem(amps, phases, y_full, mask, sim, obs)
    rows = [run_method("brent", mk, brent_kw=cfg["brent"]),
            run_method("midpoint+lsq", mk, refine_kw=cfg["lsq_refine"]),
            run_method(f"grid{cfg['grid_k']}+lsq", mk, refine_kw=cfg["lsq_refine"], grid_k=cfg["grid_k"])]
    for name, th0 in inits.items():
        rows.append(run_method(f"{name}+lsq", mk, init_theta=th0, refine_kw=cfg["lsq_refine"]))
    for r in rows:
        r.pop("trace", None)
        r.update(variant=v, nu_true=nu_true, rel_err=abs(r["nu_hat"] - nu_true) / nu_true)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_DIR / "phaseD_v1.json"))
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N variants (smoke)")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    dset = ROOT / "data" / cfg["dataset"]
    out = ROOT / "results" / (f"{cfg['config_version']}_{args.split}" + ("" if args.limit is None else "_smoke"))
    out.mkdir(parents=True, exist_ok=True)

    norm = dict(np.load(dset / "normaliser.npz")); norm["fitted_on"] = str(norm["fitted_on"])
    Ttr, Otr = load_split(dset, "train")
    Tva, Ova = load_split(dset, "val")
    Xtr, ytr = features(Otr, Ttr, norm)
    Xva, yva = features(Ova, Tva, norm)
    T, O = (Tva, Ova) if args.split == "val" else load_split(dset, "test")
    X, ytrue = features(O, T, norm)

    knn_best, knn_trials = tune_knn(cfg, Xtr, ytr, Xva, yva)
    knn = KNNInit(knn_best["k"], knn_best["ic_weight"]).fit(Xtr, ytr)
    t0 = time.perf_counter(); th_knn = knn.predict(X); knn_query_s = (time.perf_counter() - t0) / len(X)
    inits, overhead = {"knn": th_knn}, {"knn": dict(build_s=knn.build_seconds, per_query_s=knn_query_s)}
    for seed in cfg["mlp_seeds"]:
        m = load_model(ROOT / "models" / cfg["model_version"] / f"seed{seed}" / "model.pkl")
        t0 = time.perf_counter(); th = predict_theta(m, X); bt = (time.perf_counter() - t0) / len(X)
        t0 = time.perf_counter(); predict_theta(m, X[:1]); single = time.perf_counter() - t0
        inits[f"mlp_s{seed}"] = th
        overhead[f"mlp_s{seed}"] = dict(train_s=m["train_seconds"], per_query_batched_s=bt, per_query_single_s=single)

    n = len(X) if args.limit is None else args.limit
    tasks = [(v, T["amps"][O["parent"][v]], T["phases"][O["parent"][v]], O["y"][v], O["mask"][v],
              float(T["nu"][O["parent"][v]]), {k: float(a[v]) for k, a in inits.items()}, cfg) for v in range(n)]
    # checkpointing: every finished variant is appended to runs.partial.jsonl; a restart skips them.
    # Results per variant are deterministic, so resumed and uninterrupted runs give identical rows.
    partial = out / "runs.partial.jsonl"
    rows = [json.loads(l) for l in partial.read_text().splitlines()] if partial.exists() else []
    per_variant = 3 + len(inits)
    counts = {}
    for r in rows:
        counts[r["variant"]] = counts.get(r["variant"], 0) + 1
    done = {v for v, c in counts.items() if c == per_variant}
    rows = [r for r in rows if r["variant"] in done]          # drop any half-written variant
    tasks = [t for t in tasks if t[0] not in done]
    if done:
        print(f"resuming: {len(done)} variants already done, {len(tasks)} to go", flush=True)
    partial.write_text("".join(json.dumps(r) + "\n" for r in rows))
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(partial, "a") as fp:
        for k, rs in enumerate(ex.map(run_variant, tasks, chunksize=2)):
            for r in rs:
                v = r["variant"]
                r.update(n_sensors=int(O["n_sensors"][v]), noise=float(O["noise"][v]),
                         parent=int(O["parent"][v]))
                init_name = r["method"].replace("+lsq", "")
                if init_name in inits:
                    r["init_rel_err"] = float(abs(np.exp(inits[init_name][v]) - r["nu_true"]) / r["nu_true"])
            rows += rs
            fp.write("".join(json.dumps(r) + "\n" for r in rs)); fp.flush()
            if (k + 1) % 25 == 0:
                print(f"{len(done) + k + 1}/{n} variants ({time.time()-t_start:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r["variant"], r["method"]))
    keys = sorted({k for r in rows for k in r})
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    (out / "meta.json").write_text(json.dumps(dict(
        config=cfg, split=args.split, n_variants=n, knn_selected=knn_best, knn_trials=knn_trials,
        init_overhead=overhead, workers=args.workers,
        timing_note="parallel workers: use solve counts; wall times indicative only",
        finished=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
