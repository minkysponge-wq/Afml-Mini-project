"""Observation operator: fixed sensor slots x fixed times, with availability masks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .simulator import CONFIG_DIR, CallLog, SimConfig, initial_condition, interp_periodic, solve


@dataclass(frozen=True)
class ObsConfig:
    n_slots: int = 16
    times: tuple = (0.25, 0.5, 0.75, 1.0)

    @classmethod
    def from_json(cls, path=CONFIG_DIR / "simulator_v1.json"):
        o = json.loads(Path(path).read_text())["observation"]
        return cls(n_slots=o["n_sensor_slots"], times=tuple(o["times"]))

    @property
    def sensor_x(self):
        return (np.arange(self.n_slots) + 0.5) / self.n_slots


def nested_mask(n_active: int, n_slots: int = 16):
    """Evenly spaced nested subsets: 16 -> all, 8 -> every 2nd, 4 -> every 4th."""
    assert n_slots % n_active == 0
    m = np.zeros(n_slots, bool)
    m[:: n_slots // n_active] = True
    return m


def observe(amps, phases, nu, sim: SimConfig, obs: ObsConfig, log: CallLog | None = None):
    """Clean observations, shape [n_times, n_slots]."""
    u0 = initial_condition(sim.xc, amps, phases, length=sim.length)
    traj = solve(u0, nu, obs.times, sim, log)
    return interp_periodic(traj, obs.sensor_x, sim)
