import numpy as np

from blinit.inverse import TH_HI, TH_LO, brent_bracket


class Quad:
    """Stand-in for Problem: counts objective calls on a 1-D quadratic in theta."""

    def __init__(self, th_star):
        self.th_star, self.calls = th_star, 0

    def objective(self, th):
        self.calls += 1
        return (float(th) - self.th_star) ** 2


def test_bracket_finds_minimum_inside_without_fallback():
    th_star = 0.5 * (TH_LO + TH_HI) + 0.1
    q = Quad(th_star)
    th, info = brent_bracket(q, th_star - 0.05, 0.3, xatol=1e-3)
    assert abs(th - th_star) < 2e-3 and not info["fallback"]
    assert TH_LO <= info["bracket_lo"] < info["bracket_hi"] <= TH_HI


def test_bracket_falls_back_when_minimum_outside():
    th_star = TH_HI - 0.1
    q = Quad(th_star)
    th, info = brent_bracket(q, TH_LO + 0.3, 0.2, xatol=1e-3)   # bracket far below the minimum
    assert info["fallback"] and abs(th - th_star) < 2e-3


def test_bracket_clipped_to_global_bound_is_not_an_edge():
    q = Quad(TH_LO + 1e-4)                                        # minimum at the global lower bound
    th, info = brent_bracket(q, TH_LO + 0.05, 0.3, xatol=1e-3)
    assert info["bracket_lo"] == TH_LO and not info["fallback"]
    assert np.isclose(th, TH_LO, atol=3e-3)
