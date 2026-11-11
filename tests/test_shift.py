import copy

import numpy as np
import pytest

from blinit.data import (features, fit_normaliser, generate_trajectories, load_config, make_observations,
                         make_observations_masks, named_masks)
from blinit.experiments.build_shift_set import shift_integrity
from blinit.inverse import Problem
from blinit.knn import KNNInit
from blinit.observe import ObsConfig, nested_mask
from blinit.simulator import SimConfig, initial_condition, interp_periodic, solve

AMPS, PHASES, NU = np.array([0.6, 0.3, 0.2]), np.array([0.4, 1.3, 2.9]), 0.03


def test_problem_modes_reproduce_observations():
    sim, obs = SimConfig.from_json(), ObsConfig.from_json()
    modes = (2, 3, 4)
    u0 = initial_condition(sim.xc, AMPS, PHASES, modes=modes, length=sim.length)
    y = interp_periodic(solve(u0, NU, obs.times, sim), obs.sensor_x, sim)
    p = Problem(AMPS, PHASES, y, nested_mask(16), sim, obs, modes=modes)
    assert np.max(np.abs(p.residual(np.log(NU)))) < 1e-12
    # the default modes give a different trajectory, so the field is really used
    assert np.max(np.abs(Problem(AMPS, PHASES, y, nested_mask(16), sim, obs).residual(np.log(NU)))) > 1e-3


@pytest.fixture(scope="module")
def tiny_shift():
    cfg = copy.deepcopy(load_config())
    cfg["splits"] = {"train": 6, "test": 2}
    cfg["ic_seed_offset"] = {"train": 0, "test": 3_000_000}
    Ttr = generate_trajectories(cfg, "train", workers=2)
    T = generate_trajectories(cfg, "test", workers=2, modes=(2, 3, 4))
    spec = {"half_left": {"on": list(range(8, 16))}, "every4_shift": {"on": [1, 5, 9, 13]}}
    O = make_observations_masks(T, cfg, named_masks(spec))
    return cfg, Ttr, T, O


def test_named_masks_and_shift_integrity(tiny_shift):
    cfg, Ttr, T, O = tiny_shift
    assert T["modes"].tolist() == [2, 3, 4] and int(T["sim_nx"]) == SimConfig.from_json().nx
    assert len(O["parent"]) == 2 * 3 * 2 and set(O["mask_name"]) == {"half_left", "every4_shift"}
    assert O["n_sensors"][O["mask_name"] == "half_left"].tolist() == [8] * 6
    assert not O["mask"][O["mask_name"] == "half_left"][0][:8].any()
    chk = shift_integrity(T, O, cfg, set(Ttr["ic_seed"].tolist()))
    assert all(chk.values()), chk
    assert not shift_integrity(T, O, cfg, set(T["ic_seed"].tolist()))["ic_seeds_disjoint_from_base_dataset"]


def test_knn_falls_back_for_unseen_mask(tiny_shift):
    cfg, Ttr, T, O = tiny_shift
    Otr = make_observations(Ttr, cfg)
    norm = fit_normaliser(Otr)
    Xtr, ytr = features(Otr, Ttr, norm)
    X, _ = features(O, T, norm)
    pred = KNNInit(k=1, ic_weight=1.0).fit(Xtr, ytr).predict(X)
    assert pred.shape == (len(X),) and np.all(np.isfinite(pred))
    # with k=1 the answer is the label of one all-sensors library entry
    full = Otr["n_sensors"] == 16
    assert np.all(np.isin(np.round(pred, 12), np.round(ytr[full], 12)))
