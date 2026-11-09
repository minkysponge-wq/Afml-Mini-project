"""Phase C: tune (validation only) and train the MLP initialiser; report neural-only VALIDATION error.

Usage: python -m blinit.experiments.train_mlp [--config configs/phaseC_v1.json]
Outputs: models/<config_version>/{tuning.json, seed<k>/model.pkl, seed<k>/curve.csv, summary.json, figures}
The test split is not loaded here.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from blinit.data import features
from blinit.inverse import TH_HI, TH_LO
from blinit.model import predict_theta, save_model, train_mlp
from blinit.simulator import CONFIG_DIR

ROOT = Path(__file__).resolve().parents[2]


def load_split(dset_dir, split):
    T = dict(np.load(dset_dir / f"traj_{split}.npz"))
    O = dict(np.load(dset_dir / f"obs_{split}.npz"))
    O["split"] = str(O["split"])
    return T, O


def error_table(theta_hat, theta_true, O):
    rel = np.abs(np.exp(theta_hat) - np.exp(theta_true)) / np.exp(theta_true)
    tab = {"all": dict(median=float(np.median(rel)), p90=float(np.percentile(rel, 90)))}
    for s in (4, 8, 16):
        for lv in (0.0, 0.01, 0.05):
            sel = (O["n_sensors"] == s) & (O["noise"] == lv)
            if sel.any():
                tab[f"s{s}_n{lv:g}"] = dict(median=float(np.median(rel[sel])),
                                            p90=float(np.percentile(rel[sel], 90)))
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_DIR / "phaseC_v1.json"))
    cfg = json.loads(Path(ap.parse_args().config).read_text())
    dset = ROOT / "data" / cfg["dataset"]
    out = ROOT / "models" / cfg["config_version"]
    out.mkdir(parents=True, exist_ok=True)
    norm = dict(np.load(dset / "normaliser.npz"))
    norm["fitted_on"] = str(norm["fitted_on"])
    assert norm["fitted_on"] == "train"
    Ttr, Otr = load_split(dset, "train")
    Tva, Ova = load_split(dset, "val")
    Xtr, ytr = features(Otr, Ttr, norm)
    Xva, yva = features(Ova, Tva, norm)
    mc = cfg["model"]
    kw = dict(hidden=tuple(mc["hidden_layer_sizes"]), batch_size=mc["batch_size"],
              max_epochs=cfg["max_epochs"], patience=cfg["patience"])

    # tuning on seed 0, validation MSE only
    tuning = []
    for lr, alpha in itertools.product(cfg["tuning_grid"]["learning_rate_init"], cfg["tuning_grid"]["alpha"]):
        m = train_mlp(Xtr, ytr, Xva, yva, lr=lr, alpha=alpha, seed=0, **kw)
        tuning.append(dict(lr=lr, alpha=alpha, val_mse_std=m["best_val_mse_std"], epochs=m["epochs"],
                           train_seconds=m["train_seconds"]))
        print(tuning[-1], flush=True)
    best = min(tuning, key=lambda t: t["val_mse_std"])
    (out / "tuning.json").write_text(json.dumps(dict(trials=tuning, selected=best), indent=2))

    summary = dict(selected=best, n_train=len(ytr), n_val=len(yva), seeds={},
                   tuning_total_seconds=sum(t["train_seconds"] for t in tuning))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, seed in enumerate(cfg["seeds"]):
        m = train_mlp(Xtr, ytr, Xva, yva, lr=best["lr"], alpha=best["alpha"], seed=seed, **kw)
        sd = out / f"seed{seed}"
        save_model(m, sd / "model.pkl")
        np.savetxt(sd / "curve.csv", np.array(m["curve"]), delimiter=",", header="epoch,train_mse_std,val_mse_std")
        th = predict_theta(m, Xva)
        summary["seeds"][seed] = dict(epochs=m["epochs"], train_seconds=m["train_seconds"],
                                      val_mse_std=m["best_val_mse_std"], val_rel_err=error_table(th, yva, Ova))
        c = np.array(m["curve"])
        ax.plot(c[:, 0], c[:, 2], color=["#2a78d6", "#eb6834", "#1baf7a"][k], label=f"val, seed {seed}")
        ax.plot(c[:, 0], c[:, 1], color=["#2a78d6", "#eb6834", "#1baf7a"][k], lw=0.8, ls=":")
    mid = np.full_like(yva, 0.5 * (TH_LO + TH_HI))
    summary["reference_midpoint_val_rel_err"] = error_table(mid, yva, Ova)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE (standardised log nu)")
    ax.set_title("MLP training curves (dotted: train)"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=150); plt.close(fig)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for s, v in summary["seeds"].items():
        print("seed", s, "epochs", v["epochs"], "val rel err median/p90:",
              round(v["val_rel_err"]["all"]["median"], 4), round(v["val_rel_err"]["all"]["p90"], 4))
    print("midpoint guess val rel err median:", round(summary["reference_midpoint_val_rel_err"]["all"]["median"], 4))


if __name__ == "__main__":
    main()
