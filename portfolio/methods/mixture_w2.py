from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from ..radii import radius_from_transport_budget, transport_budget_from_radius
from .exact_single_ball_w2 import solve as solve_single_ball_w2
from .shared import mixture_support, source_rule_weights, source_sample_list


def solve(*, train_data: Dict[str, object], cfg, eps_radius: Optional[np.ndarray] = None, eps_cost: Optional[np.ndarray] = None, **kwargs) -> Dict[str, object]:
    if eps_radius is None and eps_cost is None:
        raise ValueError('mixture_w2.solve requires eps_radius or eps_cost')

    K = len(source_sample_list(train_data))
    t = source_rule_weights(cfg, train_data)

    if eps_radius is None:
        eps_cost_arr = np.asarray(eps_cost, dtype=float).reshape(K)
        eps_radius_arr = np.array([radius_from_transport_budget(v, 2) for v in eps_cost_arr], dtype=float)
    else:
        eps_radius_arr = np.asarray(eps_radius, dtype=float).reshape(K)
        eps_cost_arr = np.array([transport_budget_from_radius(v, 2) for v in eps_radius_arr], dtype=float)

    epsilon_mix_radius = float(np.dot(t, eps_radius_arr))
    epsilon_mix_cost = transport_budget_from_radius(epsilon_mix_radius, 2)
    nominal_samples, nominal_weights = mixture_support(cfg, train_data)
    result = solve_single_ball_w2(
        train_data=train_data,
        cfg=cfg,
        nominal_samples=nominal_samples,
        nominal_weights=nominal_weights,
        eps_cost=epsilon_mix_cost,
    )
    result['epsilon_mix_radius'] = epsilon_mix_radius
    result['epsilon_mix_cost'] = float(epsilon_mix_cost)
    result['eps_radius'] = eps_radius_arr.copy()
    result['eps_cost'] = eps_cost_arr.copy()
    result['t'] = np.array(t, dtype=float)
    return result
