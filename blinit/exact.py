"""Exact Cole-Hopf solution of u_t + u u_x = D u_xx on [0,1) for u0 = A sin(2 pi x)."""
import numpy as np
from scipy.special import ive


def cole_hopf_sine(x, t, A, nu, n_terms=None):
    D = nu / np.pi
    r = A / (4 * np.pi * D)
    if n_terms is None:
        n_terms = int(r + 40 * np.sqrt(r + 1) + 40)
    n = np.arange(1, n_terms + 1)[:, None]
    In = ive(n, r)  # exp(-r) I_n(r); the common scale cancels in the ratio
    decay = np.exp(-D * (2 * np.pi * n) ** 2 * t)
    th = 2 * np.pi * np.asarray(x)[None, :]
    num = np.sum(n * In * decay * np.sin(n * th), axis=0)
    den = ive(0, r) + 2 * np.sum(In * decay * np.cos(n * th), axis=0)
    return 8 * np.pi * D * num / den
