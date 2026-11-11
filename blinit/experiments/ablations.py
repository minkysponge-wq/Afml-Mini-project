"""T5 ablations (validation only; no new simulator calls).

Usage: python -m blinit.experiments.ablations
(a) neural-only vs neural + refinement: results/phaseD_v1_val/runs.csv, rows mlp_s0+lsq
    (init_rel_err = MLP alone, rel_err = after refinement).
(b) sensor-dropout augmentation vs none: models/phaseC_v2 vs models/phaseC_v2_noaug (16-sensor training only).
(c) with vs without the known IC coefficients as inputs: models/phaseC_v2 vs models/phaseC_v2_noic.
(b) and (c) compare the neural-only validation error stored by train_mlp in each summary.json (3 seeds).
Output: results/phaseF_ablations/ablations.json and ablations.md (three tables).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONDS = [(s, n) for s in (4, 8, 16) for n in (0.0, 0.01, 0.05)]
US_PER_STEP = 144.3


def refine_table(runs_dir, method="mlp_s0+lsq"):
    rows = [r for r in csv.DictReader(open(runs_dir / "runs.csv")) if r["method"] == method]
    tab = {}
    for key, sel in [("all", lambda r: True)] + [
            (f"s{s}_n{n:g}", (lambda s, n: lambda r: float(r["n_sensors"]) == s and float(r["noise"]) == n)(s, n))
            for s, n in CONDS]:
        rs = [r for r in rows if sel(r)]
        ini = np.array([float(r["init_rel_err"]) for r in rs]); fin = np.array([float(r["rel_err"]) for r in rs])
        cpu = np.array([float(r["solver_steps"]) for r in rs]) * US_PER_STEP * 1e-6
        tab[key] = dict(n=len(rs), neural_only_median=float(np.median(ini)), neural_only_p90=float(np.percentile(ini, 90)),
                        refined_median=float(np.median(fin)), refined_p90=float(np.percentile(fin, 90)),
                        refine_cpu_s_median=float(np.median(cpu)),
                        ratio_median=float(np.median(ini) / np.median(fin)))
    return tab


def model_table(version):
    s = json.loads((ROOT / "models" / version / "summary.json").read_text())
    keys = ["all"] + [f"s{a}_n{b:g}" for a, b in CONDS]
    tab = {}
    for k in keys:
        med = [s["seeds"][sd]["val_rel_err"][k]["median"] for sd in s["seeds"]]
        p90 = [s["seeds"][sd]["val_rel_err"][k]["p90"] for sd in s["seeds"]]
        tab[k] = dict(median_by_seed=med, median_mean=float(np.mean(med)), median_min=float(min(med)),
                      median_max=float(max(med)), p90_mean=float(np.mean(p90)))
    by_sensor = {}
    for a in (4, 8, 16):
        by_sensor[f"s{a}"] = float(np.mean([tab[f"s{a}_n{b:g}"]["median_mean"] for b in (0.0, 0.01, 0.05)]))
    return dict(n_train=s["n_train"], selected=s["selected"], epochs=[s["seeds"][sd]["epochs"] for sd in s["seeds"]],
                table=tab, median_by_sensor_count=by_sensor)


def main():
    out = ROOT / "results" / "phaseF_ablations"
    out.mkdir(parents=True, exist_ok=True)
    res = dict(a_neural_only_vs_refined=refine_table(ROOT / "results" / "phaseD_v1_val"),
               models={v: model_table(v) for v in ("phaseC_v2", "phaseC_v2_noaug", "phaseC_v2_noic")})
    (out / "ablations.json").write_text(json.dumps(res, indent=2))

    L = ["# T5 ablations (validation split, 450 variants)", "",
         "## (a) MLP alone vs MLP + least-squares refinement (seed 0)", "",
         "| condition | MLP alone, median / p90 rel. error | refined, median / p90 | median CPU-s of refinement |",
         "|---|---|---|---|"]
    for k, t in res["a_neural_only_vs_refined"].items():
        L.append(f"| {k} | {t['neural_only_median']:.3f} / {t['neural_only_p90']:.3f} | "
                 f"{t['refined_median']:.2e} / {t['refined_p90']:.2e} | {t['refine_cpu_s_median']:.2f} |")
    for title, other in (("(b) Sensor-dropout augmentation: all 4/8/16-sensor variants vs 16-sensor training only", "phaseC_v2_noaug"),
                         ("(c) Known IC coefficients as inputs vs hidden (set to 0)", "phaseC_v2_noic")):
        L += ["", f"## {title}", "",
              f"Neural-only median relative error, mean over 3 seeds [min, max]. Baseline phaseC_v2 trains on "
              f"{res['models']['phaseC_v2']['n_train']} variants, {other} on {res['models'][other]['n_train']}.", "",
              f"| condition | phaseC_v2 | {other} |", "|---|---|---|"]
        for k in res["models"]["phaseC_v2"]["table"]:
            a, b = res["models"]["phaseC_v2"]["table"][k], res["models"][other]["table"][k]
            L.append(f"| {k} | {a['median_mean']:.3f} [{a['median_min']:.3f}, {a['median_max']:.3f}] | "
                     f"{b['median_mean']:.3f} [{b['median_min']:.3f}, {b['median_max']:.3f}] |")
    (out / "ablations.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
