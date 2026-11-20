from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import cvxpy as cp

from .shared import branch_factors, pooled_support, solve_kwargs


def solve(*, train_data: Dict[str, object], cfg, nominal_samples: Optional[np.ndarray] = None, nominal_weights: Optional[np.ndarray] = None, eps_cost: Optional[float] = None, **kwargs) -> Dict[str, object]:
    if eps_cost is None:
        raise ValueError('exact_single_ball_w2.solve requires eps_cost')

    if nominal_samples is None or nominal_weights is None:
        nominal_samples, nominal_weights = pooled_support(train_data)

    support = np.asarray(nominal_samples, dtype=float)
    probs = np.asarray(nominal_weights, dtype=float).reshape(-1)
    probs = probs / probs.sum()
    M, d = support.shape

    slope_factors, tau_factors = branch_factors(cfg.rho, cfg.eta)

    x = cp.Variable(d)
    tau = cp.Variable()
    gamma = cp.Variable(nonneg=True)
    s = cp.Variable(M)

    constraints = [x >= 0, cp.sum(x) == 1]
    for ell in range(2):
        dvec = slope_factors[ell] * x
        cval = tau_factors[ell] * tau
        extra = 0.25 * cp.quad_over_lin(dvec, gamma)
        constraints.append(s >= cval + support @ dvec + extra)

    obj = cp.Minimize(gamma * float(eps_cost) + probs @ s)
    prob = cp.Problem(obj, constraints)
    prob.solve(**solve_kwargs(cfg))

    return {
        'status': prob.status,
        'objective': None if prob.value is None else float(prob.value),
        'weights': None if x.value is None else np.array(x.value, dtype=float),
        'tau': None if tau.value is None else float(tau.value),
        'eps_cost': float(eps_cost),
    }
