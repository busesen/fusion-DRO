from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class EmpiricalStats:
    source: str
    role: str
    points: np.ndarray
    probs: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    support_rows: int
    total_mass: float


def _sym(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.T)


def _sqrtm_psd(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w, v = np.linalg.eigh(_sym(mat))
    w = np.clip(w, 0.0, None)
    return v @ np.diag(np.sqrt(w + eps)) @ v.T


def _invsqrtm_psd(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w, v = np.linalg.eigh(_sym(mat))
    w = np.clip(w, eps, None)
    return v @ np.diag(1.0 / np.sqrt(w)) @ v.T


def _coalesce_support(points: np.ndarray, weights: np.ndarray, decimals: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    w = np.asarray(weights, dtype=float)
    if pts.ndim != 2:
        raise ValueError(f"points must be 2D, got shape {pts.shape}")
    if w.ndim != 1 or w.shape[0] != pts.shape[0]:
        raise ValueError(f"weights shape mismatch: points={pts.shape}, weights={w.shape}")

    key = np.round(pts, decimals=decimals)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    w_u = np.zeros(len(uniq), dtype=float)
    np.add.at(w_u, inv, w)

    keep = w_u > 0.0
    uniq = uniq[keep]
    w_u = w_u[keep]
    if uniq.shape[0] == 0:
        raise ValueError("coalesced support is empty (all weights non-positive)")
    p = w_u / float(w_u.sum())
    return uniq, p


def gelbrich_barycenter(
    means: Sequence[np.ndarray],
    covs: Sequence[np.ndarray],
    weights: Sequence[float],
    max_iter: int = 500,
    tol: float = 1e-9,
    reg: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    if len(means) != len(covs) or len(means) != len(weights):
        raise ValueError("means, covs, weights must have same length")
    if len(means) == 0:
        raise ValueError("At least one distribution is required")

    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    s = float(w.sum())
    if s <= 0:
        raise ValueError("weights must sum to positive")
    w = w / s

    mu_bar = np.zeros_like(np.asarray(means[0], dtype=float))
    for wk, mu in zip(w, means):
        mu_bar += wk * np.asarray(mu, dtype=float)

    if len(means) == 1:
        return mu_bar, _sym(np.asarray(covs[0], dtype=float)), 0, 0.0

    d = int(mu_bar.shape[0])
    eye = np.eye(d, dtype=float)
    sigma = np.zeros((d, d), dtype=float)
    for wk, cov in zip(w, covs):
        sigma += wk * _sym(np.asarray(cov, dtype=float))
    sigma = _sym(sigma) + reg * eye

    rel_err = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        sigma_reg = _sym(sigma) + reg * eye
        s_sqrt = _sqrtm_psd(sigma_reg)
        s_invsqrt = _invsqrtm_psd(sigma_reg)

        accum = np.zeros_like(sigma)
        for wk, cov in zip(w, covs):
            cov_k = _sym(np.asarray(cov, dtype=float)) + reg * eye
            inner = _sym(s_sqrt @ cov_k @ s_sqrt)
            accum += wk * _sqrtm_psd(inner)

        sigma_next = _sym(s_invsqrt @ accum @ accum @ s_invsqrt)
        denom = max(float(np.linalg.norm(sigma, ord="fro")), 1e-16)
        rel_err = float(np.linalg.norm(sigma_next - sigma, ord="fro") / denom)
        sigma = sigma_next
        if rel_err <= tol:
            break

    return mu_bar, sigma, it, rel_err


def discretize_gaussian(
    mean: np.ndarray,
    cov: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
    reg: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be >=1, got {num_samples}")
    mean = np.asarray(mean, dtype=float)
    cov = _sym(np.asarray(cov, dtype=float))
    d = int(mean.shape[0])
    cov_reg = cov + float(reg) * np.eye(d, dtype=float)
    x = rng.multivariate_normal(mean=mean, cov=cov_reg, size=int(num_samples), check_valid="warn")
    p = np.full((int(num_samples),), 1.0 / float(num_samples), dtype=float)
    return np.asarray(x, dtype=float), p


def weighted_empirical_stats(
    source_name: str,
    role: str,
    df: pd.DataFrame,
    context_cols: list[str],
    count_col: str,
) -> EmpiricalStats:
    if df.empty:
        raise ValueError(f"Source {source_name} ({role}) has no rows")

    x = df[context_cols].to_numpy(dtype=float)
    w = pd.to_numeric(df[count_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mask = w > 0.0
    x = x[mask]
    w = w[mask]

    if x.shape[0] == 0:
        raise ValueError(f"Source {source_name} ({role}) has no positive-mass rows")

    pts, p = _coalesce_support(x, w)
    mu = np.sum(p[:, None] * pts, axis=0)
    support_rows = int(pts.shape[0])
    total_mass = float(w.sum())

    xc = pts - mu[None, :]
    cov = (xc.T * p) @ xc
    cov = _sym(cov)
    return EmpiricalStats(
        source=str(source_name),
        role=role,
        points=pts,
        probs=p,
        mean=mu,
        cov=cov,
        support_rows=support_rows,
        total_mass=total_mass,
    )


def split_target_validation_test(
    target_df: pd.DataFrame,
    count_col: str,
    val_fraction: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")

    pos = target_df[pd.to_numeric(target_df[count_col], errors="coerce").fillna(0.0) > 0.0].copy()
    if pos.empty:
        raise ValueError("Target source has no positive-count rows")

    n = len(pos)
    n_val = int(np.floor(val_fraction * n))
    n_val = max(1, min(n - 1, n_val)) if n > 1 else 1

    idx = pos.index.to_numpy()
    rng.shuffle(idx)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:]

    val_df = target_df.loc[val_idx].copy()
    test_df = target_df.loc[test_idx].copy()
    return pos, val_df, test_df
