from __future__ import annotations

from typing import Dict
import numpy as np

from .config import PortfolioConfig
from .mean_patterns import make_combinatorial_overlap_blocks_pattern


def heterogeneous_covariance(direction: np.ndarray, sigma_sys: float, sigma_id: float) -> np.ndarray:
    direction = np.asarray(direction, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError('direction must be nonzero')
    u = direction / norm
    return (sigma_sys ** 2) * np.outer(u, u) + (sigma_id ** 2) * np.eye(direction.size)


def covariance_directions(K: int, d_assets: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(K, d_assets))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-12)
    return directions

def generate_populations(cfg: PortfolioConfig, seed: int) -> Dict[str, object]:
    means, groups = make_combinatorial_overlap_blocks_pattern(
        cfg.K, cfg.d_assets, cfg.block_size, cfg.stride,
        cfg.mu_high, cfg.mu_low, cfg.mu_high_growth, cfg.mu_low_shrink,
    )
    direction_seed = cfg.cov_direction_seed + (seed if cfg.cov_direction_per_run else 0)
    directions = covariance_directions(cfg.K, cfg.d_assets, direction_seed)
    covariances = np.stack([
        heterogeneous_covariance(directions[k], cfg.sigma_sys, cfg.sigma_id)
        for k in range(cfg.K)
    ])
    rng = np.random.default_rng(seed)
    samples = np.zeros((cfg.K, cfg.N_per_dist, cfg.d_assets), dtype=float)
    for k in range(cfg.K):
        samples[k] = rng.multivariate_normal(
            mean=means[k], cov=covariances[k], size=cfg.N_per_dist,
        )
    return {
        'means': means, 'covariances': covariances, 'cov_directions': directions,
        'samples': samples, 'groups': groups, 'seed': seed,
    }
