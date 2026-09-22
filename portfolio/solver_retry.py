from __future__ import annotations

from typing import Any, Callable, Dict
import time
import numpy as np
import cvxpy as cp


PORTFOLIO_SOLVER_RETRY_SCALE = 100.0

_LINEAR_DATA_KEYS = {
    'source_samples',
    'source_means',
}
_QUADRATIC_DATA_KEYS = {
    'source_covariances',
}
_LINEAR_KWARG_KEYS = {
    'nominal_samples',
    'eps',
    'eps_radius',
    'epsilon',
    'epsilon_mix_radius',
}
_QUADRATIC_KWARG_KEYS = {
    'eps_cost',
    'epsilon_mix_cost',
    'base_cost',
}


def _scale_numeric_like(value: Any, factor: float):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_scale_numeric_like(item, factor) for item in value]
    scaled = np.asarray(value, dtype=float) * float(factor)
    if scaled.ndim == 0:
        return float(scaled)
    return scaled


def _scale_train_data(train_data: Dict[str, object], factor: float) -> Dict[str, object]:
    scaled = dict(train_data)
    for key in _LINEAR_DATA_KEYS:
        if key in train_data and train_data[key] is not None:
            scaled[key] = _scale_numeric_like(train_data[key], factor)
    for key in _QUADRATIC_DATA_KEYS:
        if key in train_data and train_data[key] is not None:
            scaled[key] = _scale_numeric_like(train_data[key], factor ** 2)
    return scaled


def _scale_solver_kwargs(solver_kwargs: Dict[str, Any], factor: float, wass_order: int) -> Dict[str, Any]:
    scaled = dict(solver_kwargs)
    for key in _LINEAR_KWARG_KEYS:
        if key in scaled and scaled[key] is not None:
            scaled[key] = _scale_numeric_like(scaled[key], factor)
    quad_factor = factor ** 2
    for key in _QUADRATIC_KWARG_KEYS:
        if key in scaled and scaled[key] is not None:
            scaled[key] = _scale_numeric_like(scaled[key], quad_factor)
    return scaled


def _map_result_back(result: Dict[str, object], factor: float, wass_order: int) -> Dict[str, object]:
    mapped = dict(result)
    if mapped.get('objective') is not None:
        mapped['objective'] = float(mapped['objective']) / float(factor)
    if mapped.get('tau') is not None:
        mapped['tau'] = float(mapped['tau']) / float(factor)

    radius_factor = float(factor)
    cost_factor = float(factor) ** 2
    for key in ('eps', 'eps_radius', 'epsilon_mix_radius'):
        if key in mapped and mapped[key] is not None:
            mapped[key] = _scale_numeric_like(mapped[key], 1.0 / radius_factor)
    for key in ('eps_cost', 'epsilon_mix_cost', 'base_cost'):
        if key in mapped and mapped[key] is not None:
            mapped[key] = _scale_numeric_like(mapped[key], 1.0 / cost_factor)

    mapped['used_scaled_retry'] = True
    mapped['solver_retry_scale_factor'] = float(factor)
    return mapped


def solve_with_scaled_retry(
    solver_fn: Callable[..., Dict[str, object]],
    *,
    train_data: Dict[str, object],
    cfg,
    wass_order: int,
    scale_factor: float = PORTFOLIO_SOLVER_RETRY_SCALE,
    **solver_kwargs,
) -> Dict[str, object]:
    if wass_order != 2:
        raise ValueError('This experiment uses W2.')
    attempt_t0 = time.perf_counter()
    try:
        result = solver_fn(train_data=train_data, cfg=cfg, **solver_kwargs)
        status = str(result.get('status', '')).strip().lower()
        if str(cfg.solver).upper() == 'MOSEK' and status == 'unknown':
            raise cp.error.SolverError('MOSEK returned UNKNOWN status.')
        result = dict(result)
        result['reported_solve_time_sec'] = time.perf_counter() - attempt_t0
        result['used_scaled_retry'] = False
        return result
    except cp.error.SolverError as exc:
        if str(cfg.solver).upper() != 'MOSEK':
            raise

        scaled_train_data = _scale_train_data(train_data, scale_factor)
        scaled_solver_kwargs = _scale_solver_kwargs(solver_kwargs, scale_factor, wass_order)

        retry_t0 = time.perf_counter()
        retry_result = solver_fn(train_data=scaled_train_data, cfg=cfg, **scaled_solver_kwargs)
        retry_elapsed = time.perf_counter() - retry_t0
        retry_status = str(retry_result.get('status', '')).strip().lower()
        if retry_status == 'unknown':
            raise cp.error.SolverError(
                f"Scaled retry with factor={float(scale_factor):g} also returned UNKNOWN. Original error: {exc}"
            ) from exc

        mapped = _map_result_back(retry_result, scale_factor, wass_order)
        mapped['reported_solve_time_sec'] = retry_elapsed
        mapped['solver_retry_reason'] = str(exc)
        return mapped
