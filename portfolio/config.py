from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from math import comb, isfinite
from typing import Any


@dataclass
class PortfolioConfig:
    """Section 5.1 defaults, matched to run 20260729_094721."""
    mode: str = 'scaling'
    data: str = 'synthetic'
    wass_order: int = 2
    d_assets: int = 10
    K: int = 3
    N_per_dist: int = 10
    num_runs: int = 10
    seed_base: int = 0
    sigma_sys: float = 0.02
    sigma_id: float = 0.015
    cov_pattern: str = 'heterogeneous'
    cov_direction_seed: int = 12345
    cov_direction_per_run: bool = True
    eta: float = 0.2
    rho: float = 10.0
    N_list: list[int] = field(default_factory=lambda: [5, 10, *range(20, 101, 10)])
    K_grid: list[int] = field(default_factory=lambda: list(range(2, 21)))
    K_for_N_sweep: int = 3
    N_for_K_sweep: int = 10
    mean_pattern: str = 'combinatorial_overlap_blocks'
    block_size: int = 4
    stride: int = 2
    mu_high: float = 0.015
    mu_low: float = -0.005
    mu_high_growth: float = 0.001
    mu_low_shrink: float = 0.001
    approx_mode: str = 'both'
    run_exact: bool = True
    run_mixture: bool = True
    run_minkowski: bool = True
    fixed_delta: float = 0.01
    plot: bool = True
    out_dir: str = 'results'
    max_tuples: int | None = 200_000
    barycenter_num_samples: int = 10
    solver: str = 'MOSEK'
    mosek_license_path: str | None = None
    exact_time_limit_sec: float = 1800.0
    max_exact_constraints: int | None = 35_000
    verbose: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(',') if x.strip()]


def _parse_optional_int(raw: str) -> int | None:
    return None if raw.strip().lower() in {'none', 'off'} else int(raw)


def build_config(args: argparse.Namespace) -> PortfolioConfig:
    values = vars(args).copy()
    values['mode'] = 'scaling'
    for key in ('K_grid', 'N_list'):
        values[key] = _parse_csv_ints(values[key])
    for key in ('max_tuples', 'max_exact_constraints'):
        values[key] = _parse_optional_int(values[key])
    cfg = PortfolioConfig(**values)
    validate_config(cfg)
    return cfg


def validate_config(cfg: PortfolioConfig) -> None:
    if cfg.data != 'synthetic' or cfg.wass_order != 2:
        raise ValueError('Section 5.1 uses synthetic data and W2 ambiguity sets.')
    if cfg.mean_pattern != 'combinatorial_overlap_blocks' or cfg.cov_pattern != 'heterogeneous':
        raise ValueError('Use combinatorial mean blocks and heterogeneous covariances.')
    if cfg.d_assets < 2 or not 1 <= cfg.block_size <= cfg.d_assets:
        raise ValueError('block_size must lie between 1 and d_assets.')
    if not 1 <= cfg.stride <= min(cfg.block_size, cfg.d_assets - cfg.block_size):
        raise ValueError('stride must allow a distinct overlapping block.')
    if not cfg.K_grid or not cfg.N_list:
        raise ValueError('Both K-grid and N-list must be nonempty.')
    if len(set(cfg.K_grid)) != len(cfg.K_grid) or len(set(cfg.N_list)) != len(cfg.N_list):
        raise ValueError('Sweep values must be unique.')
    if min([*cfg.K_grid, cfg.K_for_N_sweep]) < 2:
        raise ValueError('Every reference count must be at least two.')
    if max([*cfg.K_grid, cfg.K_for_N_sweep]) > comb(cfg.d_assets, cfg.block_size):
        raise ValueError('The reference count exceeds the number of distinct mean blocks.')
    if min([*cfg.N_list, cfg.N_for_K_sweep, cfg.num_runs]) < 1:
        raise ValueError('Sample sizes and repetition count must be positive.')
    for key in ('sigma_sys', 'sigma_id', 'rho', 'fixed_delta', 'mu_high_growth', 'mu_low_shrink'):
        if not isfinite(getattr(cfg, key)) or getattr(cfg, key) < 0:
            raise ValueError(f'{key} must be finite and nonnegative.')
    if not 0 < cfg.eta < 1:
        raise ValueError('eta must lie strictly between zero and one.')
    for cap in (cfg.max_tuples, cfg.max_exact_constraints):
        if cap is not None and cap < 1:
            raise ValueError('Exact-solver caps must be positive or None.')
    cfg.K = cfg.K_for_N_sweep
    cfg.N_per_dist = cfg.N_for_K_sweep
    cfg.barycenter_num_samples = cfg.N_per_dist


def normalized_source_weights(cfg: PortfolioConfig, num_sources: int) -> list[float]:
    """Uniform reference weights for the scaling experiment."""
    if num_sources < 1:
        raise ValueError('num_sources must be positive')
    return [1.0 / num_sources] * num_sources
