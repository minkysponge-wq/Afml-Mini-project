"""Nearest-neighbour initialiser (strong classical/non-parametric baseline).

Library = TRAIN observation variants. A query is compared only with library entries that have the
same sensor-availability mask, on [standardised available readings, w * IC features], and returns
the inverse-distance-weighted mean log nu of its k nearest neighbours.
"""
from __future__ import annotations

import time

import numpy as np

from .inverse import TH_HI, TH_LO


def _split_features(X, n_read=64):
    return X[:, :n_read], X[:, n_read:2 * n_read] > 0.5, X[:, 2 * n_read:]


class KNNInit:
    def __init__(self, k=5, ic_weight=1.0):
        self.k, self.w = k, ic_weight

    def fit(self, X, theta):
        t0 = time.perf_counter()
        z, m, ic = _split_features(X)
        self.groups = {}
        for key in {tuple(r) for r in m}:
            sel = np.all(m == np.array(key), axis=1)
            cols = np.array(key)
            self.groups[key] = (np.hstack([z[sel][:, cols], self.w * ic[sel]]), theta[sel])
        self.build_seconds = time.perf_counter() - t0
        return self

    def predict(self, X):
        z, m, ic = _split_features(X)
        out = np.empty(len(X))
        for i in range(len(X)):
            key = tuple(m[i])
            if key not in self.groups:
                raise KeyError("no library entries with this sensor mask (unseen pattern)")
            L, th = self.groups[key]
            q = np.concatenate([z[i][np.array(key)], self.w * ic[i]])
            d = np.sqrt(np.sum((L - q) ** 2, axis=1))
            nn = np.argpartition(d, min(self.k, len(d) - 1))[: self.k]
            wts = 1.0 / (d[nn] + 1e-12)
            out[i] = np.sum(wts * th[nn]) / wts.sum()
        return np.clip(out, TH_LO, TH_HI)
