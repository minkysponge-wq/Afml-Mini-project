import copy

import numpy as np
import pytest

from blinit.data import (SPLITS, features, fit_normaliser, generate_trajectories, integrity_report,
                         load_config, make_observations, params_reproducible)


@pytest.fixture(scope="module")
def tiny():
    cfg = copy.deepcopy(load_config())
    cfg["splits"] = {"train": 4, "val": 2, "test": 2}
    Ts = {s: generate_trajectories(cfg, s, workers=2) for s in SPLITS}
    Os = {s: make_observations(Ts[s], cfg) for s in SPLITS}
    return cfg, Ts, Os, fit_normaliser(Os["train"])


def test_integrity_checks_pass(tiny):
    cfg, Ts, Os, norm = tiny
    chk = integrity_report(Ts, Os, norm)
    assert all(chk.values()), chk
    assert all(params_reproducible(Ts[s], cfg["base_seed"]) for s in SPLITS)


def test_leakage_is_detected(tiny):
    cfg, Ts, Os, norm = tiny
    bad = dict(Ts["val"], ic_seed=Ts["train"]["ic_seed"][:2].copy())
    assert not integrity_report(dict(Ts, val=bad), Os, norm)["ic_seeds_disjoint"]
    assert not integrity_report(Ts, Os, dict(norm, fitted_on="all"))["normaliser_fitted_on_train_only"]


def test_variant_counts_and_nesting(tiny):
    cfg, Ts, Os, _ = tiny
    O = Os["train"]
    assert len(O["parent"]) == 4 * 3 * 3   # trajectories x noise levels x sensor counts
    # same noise draw shared by nested sensor subsets of the same (trajectory, level)
    sel = (O["parent"] == 0) & (O["noise"] == 0.05)
    y4, y16 = O["y"][sel & (O["n_sensors"] == 4)][0], O["y"][sel & (O["n_sensors"] == 16)][0]
    m4 = ~np.isnan(y4)
    np.testing.assert_array_equal(y4[m4], y16[m4])
    # noise-free variants equal clean data where available
    y0 = O["y"][(O["parent"] == 0) & (O["noise"] == 0) & (O["n_sensors"] == 16)][0]
    np.testing.assert_array_equal(y0, Ts["train"]["y_clean"][0])


def test_features_shape_and_missing_encoding(tiny):
    cfg, Ts, Os, norm = tiny
    X, y = features(Os["val"], Ts["val"], norm)
    assert X.shape[1] == 64 + 64 + 6 and np.all(np.isfinite(X))
    np.testing.assert_allclose(y, np.log(Ts["val"]["nu"][Os["val"]["parent"]]))
    four = Os["val"]["n_sensors"] == 4
    assert np.all(X[four, :64][:, ~np.tile(Os["val"]["mask"][four][0], 4)] == 0.0)
