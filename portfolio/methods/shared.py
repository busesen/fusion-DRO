from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from ..config import normalized_source_weights


def branch_factors(rho: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    slope_factors = np.array([-1.0, -(1.0 + rho / eta)], dtype=float)
    tau_factors = np.array([rho, rho * (1.0 - 1.0 / eta)], dtype=float)
    return slope_factors, tau_factors


def source_sample_list(train_data: Dict[str, object]) -> List[np.ndarray]:
    raw = train_data['source_samples']
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        return [np.asarray(raw[k], dtype=float) for k in range(raw.shape[0])]
    return [np.asarray(samples, dtype=float) for samples in raw]


def source_rule_weights(cfg, train_data: Dict[str, object]) -> np.ndarray:
    num_sources = len(source_sample_list(train_data))
    return np.array(normalized_source_weights(cfg, num_sources), dtype=float)


def uniform_probs(n: int) -> np.ndarray:
    return np.ones(n, dtype=float) / float(n)


def pooled_support(train_data: Dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    samples_list = source_sample_list(train_data)
    samples = np.vstack(samples_list)
    probs = uniform_probs(samples.shape[0])
    return samples, probs


def mixture_support(cfg, train_data: Dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    samples_list = source_sample_list(train_data)
    weights = source_rule_weights(cfg, train_data)
    samples = np.vstack(samples_list)
    probs = np.concatenate([
        np.full(samples_k.shape[0], weights[k] / float(samples_k.shape[0]), dtype=float)
        for k, samples_k in enumerate(samples_list)
    ])
    probs = probs / probs.sum()
    return samples, probs


def exact_solve_kwargs(cfg) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        'solver': cfg.solver,
        'verbose': cfg.verbose,
    }
    if str(cfg.solver).upper() == 'MOSEK' and getattr(cfg, 'exact_time_limit_sec', None) is not None:
        kwargs['mosek_params'] = {
            'MSK_DPAR_OPTIMIZER_MAX_TIME': float(cfg.exact_time_limit_sec),
        }
        kwargs['accept_unknown'] = True
    return kwargs
