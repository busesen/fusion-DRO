from __future__ import annotations

from itertools import product
from typing import Dict, Optional
import time
import warnings
import math
import numpy as np
import cvxpy as cp

from .shared import branch_factors, exact_solve_kwargs, source_sample_list, uniform_probs
from .exact_single_ball_w2 import solve as solve_single_ball_w2


def solve(*, train_data: Dict[str, object], cfg, eps_cost: Optional[np.ndarray] = None, **kwargs) -> Dict[str, object]:
    if eps_cost is None:
        raise ValueError('exact_w2.solve requires eps_cost')

    t_start = time.perf_counter()
    samples_list = source_sample_list(train_data)
    K = len(samples_list)
    Ns = [samples.shape[0] for samples in samples_list]
    d = samples_list[0].shape[1]
    if K == 1:
        return solve_single_ball_w2(
            train_data=train_data,
            cfg=cfg,
            nominal_samples=samples_list[0],
            nominal_weights=uniform_probs(Ns[0]),
            eps_cost=float(np.asarray(eps_cost, dtype=float).reshape(1)[0]),
        )
    total_tuples = math.prod(Ns)
    predicted_constraints = 2 + 4 * total_tuples
    print(
        f"[exact-w2] start num_sources={K} sample_sizes={Ns} d={d} total_tuples={total_tuples} predicted_constraints={predicted_constraints}",
        flush=True,
    )
    # The exact reformulation has one block of constraints per tuple of atoms, i.e.
    # prod_k N_k tuples and 2 + 4 * prod_k N_k constraints. In our runs, the largest instances
    # that could be solved had 32,002 constraints (K = 3, N_k = 20); the next larger instance
    # (K = 4, N_k = 10, 40,002 constraints) ran out of memory, and every larger instance is
    # bigger still. To report these instances as "skipped" instead of letting the run crash or
    # spend hours before failing, the exact solve is not attempted when the size exceeds
    # max_exact_constraints (default 35,000) or max_tuples (default 200,000). With the default
    # values the tuple cap is implied by the constraint cap; it is checked first and only
    # determines the status label of the very large instances.
    if cfg.max_tuples is not None and total_tuples > cfg.max_tuples:
        return {
            'status': f'skipped_tuples_{total_tuples}',
            'objective': None,
            'weights': None,
            'tau': None,
            'num_tuples': total_tuples,
        }
    if getattr(cfg, 'max_exact_constraints', None) is not None and predicted_constraints > cfg.max_exact_constraints:
        print(
            f"[exact-w2] skipping exact solve because predicted_constraints={predicted_constraints} exceeds max_exact_constraints={cfg.max_exact_constraints}.",
            flush=True,
        )
        return {
            'status': f'skipped_constraints_{predicted_constraints}',
            'objective': None,
            'weights': None,
            'tau': None,
            'num_tuples': total_tuples,
            'predicted_constraints': predicted_constraints,
        }

    eps_cost = np.asarray(eps_cost, dtype=float).reshape(K)
    slope_factors, tau_factors = branch_factors(cfg.rho, cfg.eta)

    t0 = time.perf_counter()
    alphas = np.array(list(product(*[range(Nk) for Nk in Ns])), dtype=int)
    A = int(alphas.shape[0])
    T = 2 * A
    print(
        f"[exact-w2] built tuple index A={A} T={T} in {time.perf_counter() - t0:.3f}s",
        flush=True,
    )

    t0 = time.perf_counter()
    x = cp.Variable(d)
    tau = cp.Variable()
    lambdas = cp.Variable(K, nonneg=True)
    gamma = [cp.Variable(Nk) for Nk in Ns]
    w = cp.Variable((T * K, d))
    print(
        f"[exact-w2] created variables in {time.perf_counter() - t0:.3f}s",
        flush=True,
    )

    constraints = [x >= 0, cp.sum(x) == 1]

    def t_index(alpha_idx: int, branch_idx: int) -> int:
        return 2 * alpha_idx + branch_idx

    t0 = time.perf_counter()
    for alpha_idx in range(A):
        alpha = alphas[alpha_idx]
        for ell in range(2):
            t = t_index(alpha_idx, ell)
            lhs = tau_factors[ell] * tau
            rhs = 0
            w_sum = 0
            for k in range(K):
                wk = w[t * K + k, :]
                j = int(alpha[k])
                lhs = lhs + samples_list[k][j, :] @ wk + 0.25 * cp.quad_over_lin(wk, lambdas[k])
                rhs = rhs + gamma[k][j]
                w_sum = w_sum + wk
            constraints.append(lhs <= rhs)
            constraints.append(w_sum == slope_factors[ell] * x)
    print(
        f"[exact-w2] built constraints count={len(constraints)} in {time.perf_counter() - t0:.3f}s",
        flush=True,
    )

    t0 = time.perf_counter()
    gamma_term = cp.sum([cp.sum(gamma[k]) / float(Ns[k]) for k in range(K)])
    obj = cp.Minimize(cp.sum(cp.multiply(lambdas, eps_cost)) + gamma_term)
    prob = cp.Problem(obj, constraints)
    print(
        f"[exact-w2] assembled cvxpy problem in {time.perf_counter() - t0:.3f}s",
        flush=True,
    )
    if str(cfg.solver).upper() == 'MOSEK':
        print('[exact-w2] canonicalizing/stuffing problem for MOSEK', flush=True)
        t0 = time.perf_counter()
        prob.get_problem_data(cp.MOSEK)
        print(
            f"[exact-w2] canonicalization/stuffing finished in {time.perf_counter() - t0:.3f}s",
            flush=True,
        )
    caught_warnings = []
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('always')
            print('[exact-w2] calling solver', flush=True)
            t0 = time.perf_counter()
            prob.solve(**exact_solve_kwargs(cfg))
            print(
                f"[exact-w2] solver returned in {time.perf_counter() - t0:.3f}s with status={prob.status}",
                flush=True,
            )
    except cp.error.SolverError:
        if x.value is None:
            limit = getattr(cfg, 'exact_time_limit_sec', None)
            print(f"[exact-w2] couldn't find a usable solution within the time limit ({limit} s).")
            return {
                'status': 'no_usable_solution_within_time_limit',
                'objective': None,
                'weights': None,
                'tau': None,
                'num_tuples': total_tuples,
                'time_limit_sec': limit,
            }
        raise

    if any('time limit' in str(w.message).lower() for w in caught_warnings):
        print(f"[exact-w2] time limit reached; reporting last MOSEK solution with status={prob.status}.")

    return {
        'status': prob.status,
        'objective': None if prob.value is None else float(prob.value),
        'weights': None if x.value is None else np.array(x.value, dtype=float),
        'tau': None if tau.value is None else float(tau.value),
        'num_tuples': total_tuples,
        'predicted_constraints': predicted_constraints,
        'time_limit_sec': getattr(cfg, 'exact_time_limit_sec', None),
        'build_time_sec': time.perf_counter() - t_start,
    }
