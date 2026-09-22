from __future__ import annotations

from typing import Dict
import numpy as np
import cvxpy as cp


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def matrix_sqrt_psd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vals, vecs = np.linalg.eigh(_sym(A))
    vals = np.clip(vals, eps, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def matrix_invsqrt_psd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vals, vecs = np.linalg.eigh(_sym(A))
    vals = np.clip(vals, eps, None)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def compute_w2_gaussian_barycenter(
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> Dict[str, object]:
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    bary_mean = weights @ means

    if np.allclose(covariances, covariances[0][None, :, :]):
        return {
            'mean': bary_mean,
            'covariance': covariances[0].copy(),
            'method': 'equal_covariance_shortcut',
            'num_iter': 0,
        }

    Sigma = np.average(covariances, axis=0, weights=weights)
    Sigma = _sym(Sigma)
    for it in range(1, max_iter + 1):
        sqrt_S = matrix_sqrt_psd(Sigma)
        invsqrt_S = matrix_invsqrt_psd(Sigma)
        accum = np.zeros_like(Sigma)
        for wk, Ck in zip(weights, covariances):
            accum += wk * matrix_sqrt_psd(sqrt_S @ Ck @ sqrt_S)
        Sigma_new = invsqrt_S @ (accum @ accum) @ invsqrt_S
        Sigma_new = _sym(Sigma_new)
        if np.linalg.norm(Sigma_new - Sigma, ord='fro') <= tol * max(1.0, np.linalg.norm(Sigma, ord='fro')):
            Sigma = Sigma_new
            break
        Sigma = Sigma_new

    return {
        'mean': bary_mean,
        'covariance': Sigma,
        'method': 'alvarez_fixed_point',
        'num_iter': it,
    }


def sample_gaussian_barycenter(mean: np.ndarray, covariance: np.ndarray, num_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean=mean, cov=covariance, size=num_samples)

def discrete_transport_cost(x_src, w_src, x_tgt, w_tgt, solver='MOSEK') -> float:
    """Squared empirical W2 distance."""
    diff = x_src[:, None, :] - x_tgt[None, :, :]
    cost = np.sum(diff * diff, axis=2)
    transport = cp.Variable(cost.shape, nonneg=True)
    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(cost, transport))),
        [cp.sum(transport, axis=1) == w_src, cp.sum(transport, axis=0) == w_tgt],
    )
    problem.solve(solver=solver, verbose=False)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or problem.value is None:
        raise RuntimeError(f'Radius-calibration transport problem failed: {problem.status}')
    return float(problem.value)
