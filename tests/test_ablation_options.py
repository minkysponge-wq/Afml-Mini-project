import numpy as np

from blinit.experiments.train_mlp import filter_variants, zero_ic


def _obs():
    n = 6
    return dict(parent=np.arange(n), y=np.zeros((n, 4, 16)), mask=np.ones((n, 16), bool),
                noise=np.array([0, 0.01, 0.05] * 2), n_sensors=np.array([4, 8, 16, 16, 4, 8]), split="train")


def test_filter_variants_keeps_only_matches():
    O = _obs()
    F = filter_variants(O, {"n_sensors": [16]})
    assert F["parent"].tolist() == [2, 3] and F["y"].shape == (2, 4, 16) and F["split"] == "train"
    assert filter_variants(O, None) is O
    F2 = filter_variants(O, {"n_sensors": [4, 8], "noise": [0.05]})
    assert F2["parent"].tolist() == [5]


def test_zero_ic_only_touches_last_six_columns():
    X = np.arange(134 * 2, dtype=float).reshape(2, 134) + 1
    Z = zero_ic(X)
    assert np.all(Z[:, -6:] == 0) and np.array_equal(Z[:, :-6], X[:, :-6]) and X[0, -1] != 0
