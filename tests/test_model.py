import numpy as np

from blinit.inverse import TH_HI, TH_LO
from blinit.model import predict_theta, train_mlp


def test_mlp_learns_simple_map_and_clips():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (600, 5))
    y = TH_LO + (TH_HI - TH_LO) * (0.5 + 0.4 * X[:, 0])
    m = train_mlp(X[:500], y[:500], X[500:], y[500:], hidden=(16,), lr=3e-3, max_epochs=200, patience=20)
    pred = predict_theta(m, X[500:])
    assert np.sqrt(np.mean((pred - y[500:]) ** 2)) < 0.1 * (TH_HI - TH_LO)
    far = predict_theta(m, np.full((1, 5), 50.0))
    assert TH_LO <= far[0] <= TH_HI
    assert m["epochs"] <= 200 and len(m["curve"]) == m["epochs"]
