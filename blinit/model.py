"""MLP initialiser: training with validation early stopping, saving and inference."""
from __future__ import annotations

import copy
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor

from .inverse import TH_HI, TH_LO


def train_mlp(Xtr, ytr, Xva, yva, hidden=(64, 64), lr=1e-3, alpha=1e-4, batch_size=64,
              max_epochs=400, patience=30, seed=0):
    """Epoch-wise partial_fit with early stopping on the given validation set."""
    y_mean, y_std = float(ytr.mean()), float(ytr.std())
    zt, zv = (ytr - y_mean) / y_std, (yva - y_mean) / y_std
    net = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=alpha,
                       learning_rate_init=lr, batch_size=batch_size, random_state=seed, max_iter=1,
                       shuffle=True)
    rng = np.random.default_rng(seed)
    best, best_val, wait, curve = None, np.inf, 0, []
    t0 = time.perf_counter()
    for ep in range(max_epochs):
        idx = rng.permutation(len(Xtr))
        net.partial_fit(Xtr[idx], zt[idx])
        tr = float(np.mean((net.predict(Xtr) - zt) ** 2))
        va = float(np.mean((net.predict(Xva) - zv) ** 2))
        curve.append((ep + 1, tr, va))
        if va < best_val - 1e-6:
            best, best_val, wait = copy.deepcopy(net), va, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return dict(net=best, y_mean=y_mean, y_std=y_std, curve=curve, best_val_mse_std=best_val,
                epochs=len(curve), train_seconds=time.perf_counter() - t0)


def predict_theta(model, X):
    """Predicted log nu, clipped to the parameter bounds."""
    z = model["net"].predict(X)
    return np.clip(z * model["y_std"] + model["y_mean"], TH_LO, TH_HI)


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({k: v for k, v in model.items()}, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
