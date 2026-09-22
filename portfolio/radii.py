from __future__ import annotations

import numpy as np
from .barycenters import discrete_transport_cost


def transport_budget_from_radius(radius: float, wass_order: int = 2) -> float:
    if wass_order != 2:
        raise ValueError('This experiment uses W2.')
    return float(radius) ** 2


def radius_from_transport_budget(budget: float, wass_order: int = 2) -> float:
    if wass_order != 2:
        raise ValueError('This experiment uses W2.')
    return float(np.sqrt(max(budget, 0.0)))


def barycenter_radii_discrete(source_samples, bary_atoms, bary_weights, delta, solver='MOSEK'):
    base_cost = np.array([
        discrete_transport_cost(
            samples, np.ones(len(samples)) / len(samples),
            bary_atoms, bary_weights, solver=solver,
        )
        for samples in source_samples
    ], dtype=float)
    return {
        'eps_cost': (1.0 + delta) ** 2 * base_cost,
        'eps_radius': (1.0 + delta) * np.sqrt(np.maximum(base_cost, 0.0)),
        'base_cost': base_cost,
    }
