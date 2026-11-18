"""The data holder's releases of Section 5.3 and Appendix C: noisy moments, a noisy histogram and their radii.

Both statistics are computed from the N hidden records, the moments from the binned numerical features
(credit_data.bin_midpoints). The moments are released with the analytic Gaussian mechanism under
(0.5, 1e-6)-differential privacy and the histogram with the Laplace mechanism under (0.5, 0)-differential privacy,
so the two releases are jointly (1, 1e-6)-differentially private. Each radius makes its ambiguity set contain the
hidden empirical distribution with probability 95%.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from scipy.special import log_ndtr
from scipy.stats import binom, chi2

from credit_data import N_CELLS, N_NUM, N_PHI, cell, phi

EPS_MOMENTS, DELTA_MOMENTS, EPS_HISTOGRAM, CONFIDENCE = 0.5, 1e-6, 0.5, 0.95
RELEASE_SEED = 20261401
MOMENT_STREAM, HISTOGRAM_STREAM = (8, 2), (8, 1)     # seed entropy: [RELEASE_SEED, simulation, *stream]
# Replace-one L2 sensitivity of the mean of phi over N records is PHI_DIAM / N: the coordinates of phi range over
# intervals of length 2 (z_k, z_i z_j for i < j, z_k y), 1 (z_k^2, y), on z in [-1, 1]^14 and y in {0, 1}.
PHI_DIAM = math.sqrt(2 * N_NUM * 4 + N_NUM * 1 + (N_NUM * (N_NUM - 1) // 2) * 4 + 1)   # sqrt(491)


def gaussian_delta(sigma: float, eps: float, sens: float) -> float:
    """delta(eps) of the Gaussian mechanism N(0, sigma^2) with L2 sensitivity sens (Balle and Wang 2018, Thm 8)."""
    a, b = sens / (2.0 * sigma), eps * sigma / sens
    return math.exp(log_ndtr(a - b)) - math.exp(eps + log_ndtr(-a - b))


def analytic_gaussian_sigma(eps: float, delta: float, sens: float) -> float:
    """Smallest sigma (to relative 1e-12, rounded up) with gaussian_delta(sigma) <= delta, by bisection."""
    lo, hi = sens * 1e-6, sens
    while gaussian_delta(hi, eps, sens) > delta:
        lo, hi = hi, 2.0 * hi
    while hi / lo - 1.0 >= 1e-12:
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if gaussian_delta(mid, eps, sens) > delta else (lo, mid)
    return hi


def project_simplex(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    k = np.nonzero(u - css / np.arange(1, len(v) + 1) > 0)[0][-1]
    return np.maximum(v - css[k] / (k + 1), 0.0)


@lru_cache(maxsize=None)
def centered_l1_quantile(conf: float = CONFIDENCE, n_mc: int = 4_000_000, seed: int = 20261022) -> float:
    """Upper confidence bound (level 1 - 1e-6) on the conf-quantile of ||e - mean(e) 1||_1 / 2 for e ~ Lap(1)^432,
    estimated from n_mc Monte Carlo draws."""
    rng = np.random.default_rng(seed)
    s = np.concatenate([0.5 * np.abs(e - e.mean(1, keepdims=True)).sum(1)
                        for e in (rng.laplace(0.0, 1.0, (50_000, N_CELLS)) for _ in range(n_mc // 50_000))])
    k = int(binom.ppf(1.0 - 1e-6, n_mc, conf)) + 1
    return float(np.sort(s)[k - 1])


def release(hidden: dict, sim: int) -> dict:
    """Noisy moments m_hat with radius r_M and noisy histogram p_hat with total variation radius r_TV."""
    n = len(hidden["y"])
    m = phi(hidden["z_binned"], hidden["y"]).mean(0)
    sigma = analytic_gaussian_sigma(EPS_MOMENTS, DELTA_MOMENTS, PHI_DIAM / n)
    m_hat = m + sigma * np.random.default_rng([RELEASE_SEED, sim, *MOMENT_STREAM]).standard_normal(N_PHI)
    p = np.bincount(cell(hidden["cats"], hidden["y"]), minlength=N_CELLS) / n
    b = 2.0 / (n * EPS_HISTOGRAM)        # Laplace scale on the proportions: replacing one record moves two counts
    p_hat = project_simplex(p + np.random.default_rng([RELEASE_SEED, sim, *HISTOGRAM_STREAM]).laplace(0.0, b, N_CELLS))
    return {"m_hat": m_hat, "r_M": float(sigma * np.sqrt(chi2.ppf(CONFIDENCE, N_PHI))),
            "p_hat": p_hat, "r_TV": float(min(1.0, b * centered_l1_quantile()))}


def certify_nonempty(rel: dict, candidates: dict) -> dict:
    """Certifies that the intersection of the two ambiguity sets is nonempty by a distribution q on finitely many
    candidate points of Xi: the distinct candidate records (binned numerical features) combined with both labels.
    Maximizes the common relative slack s of the two constraints, then verifies the solution in float64."""
    z, cats = candidates["z_binned"], candidates["cats"]
    _, keep = np.unique(np.concatenate([z, cats], 1), axis=0, return_index=True)
    z, cats = np.tile(z[keep], (2, 1)), np.tile(cats[keep], (2, 1))
    y = np.repeat([0, 1], len(keep))
    Phi = phi(z, y)
    H = sp.csr_matrix((np.ones(len(y)), (np.arange(len(y)), cell(cats, y))), shape=(len(y), N_CELLS))
    q, s = cp.Variable(len(y), nonneg=True), cp.Variable()
    cons = [cp.sum(q) == 1, cp.norm(Phi.T @ q - rel["m_hat"]) <= (1 - s) * rel["r_M"],
            0.5 * cp.norm1(H.T @ q - rel["p_hat"]) <= (1 - s) * rel["r_TV"]]
    cp.Problem(cp.Maximize(s), cons).solve(solver=cp.CLARABEL)
    qv = np.maximum(q.value, 0.0)
    qv /= qv.sum()
    dist_M = float(np.linalg.norm(Phi.T @ qv - rel["m_hat"]))
    dist_TV = float(0.5 * np.abs(H.T @ qv - rel["p_hat"]).sum())
    return {"certified": dist_M <= rel["r_M"] and dist_TV <= rel["r_TV"], "support_size": int((qv > 0).sum()),
            "dist_M": dist_M, "dist_TV": dist_TV}
