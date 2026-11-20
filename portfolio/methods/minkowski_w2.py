from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import cvxpy as cp

from .shared import branch_factors, source_rule_weights, source_sample_list, uniform_probs, solve_kwargs
from .exact_single_ball_w2 import solve as solve_single_ball_w2


def solve(*, train_data: Dict[str, object], cfg, eps_cost: Optional[np.ndarray] = None, **kwargs) -> Dict[str, object]:
    if eps_cost is None:
        raise ValueError('minkowski_w2.solve requires eps_cost')

    samples_list = source_sample_list(train_data)
    K = len(samples_list)
    Ns = [samples.shape[0] for samples in samples_list]
    d = samples_list[0].shape[1]
    eps_cost = np.asarray(eps_cost, dtype=float).reshape(K)
    if K == 1:
        return solve_single_ball_w2(
            train_data=train_data,
            cfg=cfg,
            nominal_samples=samples_list[0],
            nominal_weights=uniform_probs(Ns[0]),
            eps_cost=float(eps_cost[0]),
        )

    t = source_rule_weights(cfg, train_data)
    slope_factors, tau_factors = branch_factors(cfg.rho, cfg.eta)

    x = cp.Variable(d)
    tau = cp.Variable()
    gamma = cp.Variable(K, nonneg=True)
    s = [cp.Variable(Nk) for Nk in Ns]

    constraints = [x >= 0, cp.sum(x) == 1]
    for k in range(K):
        for ell in range(2):
            dvec = t[k] * slope_factors[ell] * x
            cval = t[k] * tau_factors[ell] * tau
            extra = 0.25 * cp.quad_over_lin(dvec, gamma[k])
            constraints.append(s[k] >= cval + samples_list[k] @ dvec + extra)

    sample_term = cp.sum([cp.sum(s[k]) / float(Ns[k]) for k in range(K)])
    obj = cp.Minimize(cp.sum(cp.multiply(gamma, eps_cost)) + sample_term)
    prob = cp.Problem(obj, constraints)
    prob.solve(**solve_kwargs(cfg))

    return {
        'status': prob.status,
        'objective': None if prob.value is None else float(prob.value),
        'weights': None if x.value is None else np.array(x.value, dtype=float),
        'tau': None if tau.value is None else float(tau.value),
        'eps_cost': eps_cost.copy(),
        't': np.array(t, dtype=float),
    }
