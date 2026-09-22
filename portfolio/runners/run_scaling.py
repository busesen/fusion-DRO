from __future__ import annotations

import copy
import time
from typing import Dict, List
import numpy as np
import pandas as pd

from ..barycenters import compute_w2_gaussian_barycenter, sample_gaussian_barycenter
from ..data_gen import generate_populations
from ..radii import barycenter_radii_discrete
from ..methods.exact_msdro_w2 import solve as solve_exact_w2
from ..methods.mixture_w2 import solve as solve_mixture_w2
from ..methods.minkowski_w2 import solve as solve_minkowski_w2
from ..methods.affine_w2 import solve as solve_affine_w2
from ..methods.static_w2 import solve as solve_static_w2
from ..methods.shared import source_rule_weights
from ..solver_retry import solve_with_scaled_retry


def _all_source_train_data(pop_data) -> Dict[str, object]:
    means = np.asarray(pop_data['means'], dtype=float)
    covariances = np.asarray(pop_data['covariances'], dtype=float)
    samples = np.asarray(pop_data['samples'], dtype=float)
    return {
        'source_means': means,
        'source_covariances': covariances,
        'source_samples': samples,
        'source_indices': list(range(means.shape[0])),
        'groups': pop_data['groups'],
    }


def _prefix_pop_data(pop_data, K_total: int) -> Dict[str, object]:
    out = {
        'means': np.asarray(pop_data['means'], dtype=float)[:K_total],
        'covariances': np.asarray(pop_data['covariances'], dtype=float)[:K_total],
        'samples': np.asarray(pop_data['samples'], dtype=float)[:K_total],
        'groups': list(pop_data['groups'][:K_total]),
        'seed': pop_data['seed'],
    }
    cov_dirs = pop_data.get('cov_directions')
    if cov_dirs is None:
        out['cov_directions'] = None
    else:
        out['cov_directions'] = np.asarray(cov_dirs, dtype=float)[:K_total]
    return out

def _job_cfg(cfg, *, K_total: int, N_per_dist: int):
    job_cfg = copy.deepcopy(cfg)
    job_cfg.K = int(K_total)
    job_cfg.N_per_dist = int(N_per_dist)
    job_cfg.barycenter_num_samples = int(N_per_dist)
    return job_cfg


def _empirical_mean_covariances(source_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_samples = np.asarray(source_samples, dtype=float)
    means = np.mean(source_samples, axis=1)
    covariances = []
    for k in range(source_samples.shape[0]):
        centered = source_samples[k] - means[k]
        cov = centered.T @ centered / float(source_samples[k].shape[0])
        covariances.append(cov)
    return means, np.stack(covariances, axis=0)


def _sampled_w2_gaussian_barycenter_center(train_data, cfg):
    source_samples = np.asarray(train_data['source_samples'], dtype=float)
    weights = source_rule_weights(cfg, train_data)
    means, covariances = _empirical_mean_covariances(source_samples)
    bary = compute_w2_gaussian_barycenter(means, covariances, weights)
    center_samples = sample_gaussian_barycenter(
        mean=np.asarray(bary['mean'], dtype=float),
        covariance=np.asarray(bary['covariance'], dtype=float),
        num_samples=int(cfg.barycenter_num_samples),
        seed=int(cfg.seed_base),
    )
    center_weights = np.ones(center_samples.shape[0], dtype=float) / float(center_samples.shape[0])
    return center_samples, center_weights, bary

def _build_per_source_radii_from_fixed_center(train_data, cfg, delta, center_samples, center_weights, center_meta):
    """Use individual radii for all methods, with one shared calibration center."""
    radii = barycenter_radii_discrete(
        source_samples=np.asarray(train_data['source_samples'], dtype=float),
        bary_atoms=np.asarray(center_samples, dtype=float),
        bary_weights=np.asarray(center_weights, dtype=float),
        delta=delta,
        solver=cfg.solver,
    )
    return {
        'eps_cost': radii['eps_cost'],
        'eps': radii['eps_radius'],
        'base_cost_full': radii['base_cost'],
        'center_meta': center_meta,
        'sampled_center': center_samples,
    }


def _solve_method(method, train_data, cfg, radii):
    solvers = {
        'exact': solve_exact_w2,
        'affine': solve_affine_w2,
        'static': solve_static_w2,
        'mixture': solve_mixture_w2,
        'minkowski': solve_minkowski_w2,
    }
    budget = {'eps_radius': radii['eps']} if method == 'mixture' else {'eps_cost': radii['eps_cost']}
    return solve_with_scaled_retry(
        solvers[method], train_data=train_data, cfg=cfg, wass_order=2, **budget,
    )


def _scaling_methods(cfg):
    methods = []
    if cfg.run_exact:
        methods.append('exact')
    if cfg.run_mixture:
        methods.append('mixture')
    if cfg.run_minkowski:
        methods.append('minkowski')
    if cfg.approx_mode in {'both', 'static'}:
        methods.append('static')
    if cfg.approx_mode in {'both', 'affine'}:
        methods.append('affine')
    return methods


def run_scaling(cfg) -> pd.DataFrame:
    if cfg.data != 'synthetic':
        raise ValueError('scaling currently supports only data=synthetic')
    if cfg.wass_order != 2:
        raise ValueError('scaling currently supports only wass_order=2')

    records: List[Dict[str, object]] = []
    K_grid = cfg.K_grid or [cfg.K]
    N_grid = cfg.N_list or [cfg.N_per_dist]
    K_max = max(K_grid) if K_grid else int(cfg.K)

    for run in range(cfg.num_runs):
        seed = cfg.seed_base + run

        # K sweep: generate one full K_max family and then use prefixes so that
        # increasing K truly adds constraints rather than regenerating a new family.
        if K_grid:
            full_cfg = _job_cfg(cfg, K_total=K_max, N_per_dist=int(cfg.N_for_K_sweep))
            pop_data_full = generate_populations(full_cfg, seed=seed)
            train_data_full = _all_source_train_data(pop_data_full)
            center_samples, center_weights, center_meta = _sampled_w2_gaussian_barycenter_center(train_data_full, full_cfg)

            for K_total in K_grid:
                sweep_axis = 'K'
                N_per_dist = int(cfg.N_for_K_sweep)
                job_cfg = _job_cfg(cfg, K_total=int(K_total), N_per_dist=N_per_dist)
                pop_data = _prefix_pop_data(pop_data_full, int(K_total))
                train_data = _all_source_train_data(pop_data)
                print(f"[scaling] run={run + 1}/{cfg.num_runs} sweep={sweep_axis} total_K={K_total} N_per_dist={N_per_dist} seed={seed}")
                radii = _build_per_source_radii_from_fixed_center(
                    train_data, job_cfg, delta=cfg.fixed_delta,
                    center_samples=center_samples, center_weights=center_weights, center_meta=center_meta,
                )
                radius_scheme = 'per_source_gelbrich'

                for method in _scaling_methods(job_cfg):
                    print(
                        f"[scaling] solving method={method} "
                        f"sweep={sweep_axis} K={K_total} "
                        f"N_per_dist={N_per_dist} seed={seed} delta={cfg.fixed_delta} radius_scheme={radius_scheme}"
                    )
                    t0 = time.perf_counter()
                    result = _solve_method(method, train_data, job_cfg, radii)
                    elapsed = float(result.get('reported_solve_time_sec', time.perf_counter() - t0))
                    objective = result.get('objective')
                    status = result.get('status')
                    print(
                        f"[scaling] finished method={method} "
                        f"sweep={sweep_axis} K={K_total} "
                        f"status={status} time={elapsed:.3f}s"
                    )
                    records.append({
                        'run': run,
                        'seed': seed,
                        'sweep_axis': sweep_axis,
                        'K_total': int(K_total),
                        'N_per_dist': N_per_dist,
                        'K_for_N_sweep': int(cfg.K_for_N_sweep),
                        'N_for_K_sweep': int(cfg.N_for_K_sweep),
                        'method': method,
                        'status': status,
                        'solve_time_sec': elapsed,
                        'objective': np.nan if objective is None else float(objective),
                        'radius_scheme': radius_scheme,
                    })

        # N sweep: for each N, build one K-fixed family; every method uses the
        # per-source radii eps_k = (1+delta) * W2(P_k, P_center).
        if N_grid:
            for N_per_dist in N_grid:
                sweep_axis = 'N'
                K_total = int(cfg.K_for_N_sweep)
                job_cfg = _job_cfg(cfg, K_total=K_total, N_per_dist=int(N_per_dist))
                pop_data = generate_populations(job_cfg, seed=seed)
                train_data = _all_source_train_data(pop_data)
                center_samples, center_weights, center_meta = _sampled_w2_gaussian_barycenter_center(train_data, job_cfg)
                print(f"[scaling] run={run + 1}/{cfg.num_runs} sweep={sweep_axis} total_K={K_total} N_per_dist={N_per_dist} seed={seed}")
                radii = _build_per_source_radii_from_fixed_center(
                    train_data, job_cfg, delta=cfg.fixed_delta,
                    center_samples=center_samples, center_weights=center_weights, center_meta=center_meta,
                )
                radius_scheme = 'per_source_gelbrich'

                for method in _scaling_methods(job_cfg):
                    print(
                        f"[scaling] solving method={method} "
                        f"sweep={sweep_axis} K={K_total} "
                        f"N_per_dist={N_per_dist} seed={seed} delta={cfg.fixed_delta} radius_scheme={radius_scheme}"
                    )
                    t0 = time.perf_counter()
                    result = _solve_method(method, train_data, job_cfg, radii)
                    elapsed = float(result.get('reported_solve_time_sec', time.perf_counter() - t0))
                    objective = result.get('objective')
                    status = result.get('status')
                    print(
                        f"[scaling] finished method={method} "
                        f"sweep={sweep_axis} K={K_total} "
                        f"status={status} time={elapsed:.3f}s"
                    )
                    records.append({
                        'run': run,
                        'seed': seed,
                        'sweep_axis': sweep_axis,
                        'K_total': K_total,
                        'N_per_dist': int(N_per_dist),
                        'K_for_N_sweep': int(cfg.K_for_N_sweep),
                        'N_for_K_sweep': int(cfg.N_for_K_sweep),
                        'method': method,
                        'status': status,
                        'solve_time_sec': elapsed,
                        'objective': np.nan if objective is None else float(objective),
                        'radius_scheme': radius_scheme,
                    })

    return pd.DataFrame(records)
