from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import cvxpy as cp

from .shared import branch_factors, source_rule_weights, source_sample_list, uniform_probs, solve_kwargs
from .exact_single_ball_w2 import solve as solve_single_ball_w2


def solve(*, train_data: Dict[str, object], cfg, eps_cost: Optional[np.ndarray] = None, psd_shift: float = 1e-8, **kwargs) -> Dict[str, object]:
    if eps_cost is None:
        raise ValueError('affine_w2.solve requires eps_cost')

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
    eps_cost = np.asarray(eps_cost, dtype=float).reshape(K)
    t = source_rule_weights(cfg, train_data)
    slope_factors, tau_factors = branch_factors(cfg.rho, cfg.eta)

    x = cp.Variable(d)
    tau = cp.Variable()
    a = cp.Variable((K, d))
    gamma = cp.Variable(K, nonneg=True)
    s = [cp.Variable(Nk) for Nk in Ns]
    A = [cp.Variable((d, d), symmetric=True) for _ in range(K)]

    constraints = [x >= 0, cp.sum(x) == 1, cp.sum(a, axis=0) == 0]
    constraints.append(sum(A) == 0)

    eye_d = np.eye(d)
    for k in range(K):
        constraints.append(gamma[k] >= psd_shift)
        for ell in range(2):
            dvec = t[k] * slope_factors[ell] * x + a[k, :]
            cval = t[k] * tau_factors[ell] * tau
            for i in range(Ns[k]):
                xi = samples_list[k][i, :]
                P = gamma[k] * eye_d - A[k]
                v = dvec + 2.0 * gamma[k] * xi
                scalar = s[k][i] - cval + gamma[k] * float(xi @ xi)
                block = cp.bmat([
                    [P, 0.5 * cp.reshape(v, (d, 1), order='F')],
                    [0.5 * cp.reshape(v, (1, d), order='F'), cp.reshape(scalar, (1, 1), order='F')],
                ])
                constraints.append(block >> 0)

    sample_term = cp.sum([cp.sum(s[k]) / float(Ns[k]) for k in range(K)])
    obj = cp.Minimize(cp.sum(cp.multiply(gamma, eps_cost)) + sample_term)
    prob = cp.Problem(obj, constraints)
    prob.solve(**solve_kwargs(cfg))

    return {
        'status': prob.status,
        'objective': None if prob.value is None else float(prob.value),
        'weights': None if x.value is None else np.array(x.value, dtype=float),
        'tau': None if tau.value is None else float(tau.value),
        'A_value': None if any(mat.value is None for mat in A) else np.stack([np.array(mat.value, dtype=float) for mat in A], axis=0),
    }
