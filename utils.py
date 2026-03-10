import ast
import json
import os
import random
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def str2bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean from: {x}")


def _parse_maybe_literal(raw: Any) -> Any:
    if isinstance(raw, (list, tuple, dict)):
        return raw
    s = str(raw).strip()
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    return [v.strip() for v in s.split(",") if v.strip()]


def parse_float_list(raw: Any) -> List[float]:
    val = _parse_maybe_literal(raw)
    if isinstance(val, (int, float, str)):
        val = [val]
    return [float(v) for v in val]


def parse_int_list(raw: Any) -> List[int]:
    val = _parse_maybe_literal(raw)
    if isinstance(val, (int, float, str)):
        val = [val]
    return [int(v) for v in val]


def parse_str_list(raw: Any) -> List[str]:
    val = _parse_maybe_literal(raw)
    if isinstance(val, str):
        val = [val]
    return [str(v) for v in val]


def split_pooled_params(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    beta = x[:2]
    mu = x[2:3]
    return beta, mu


def pack_pooled_params(beta: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    return torch.cat([beta.reshape(-1), mu.reshape(-1)], dim=0)


def project_pooled_params(x: torch.Tensor) -> torch.Tensor:
    beta, mu = split_pooled_params(x)
    beta = torch.clamp(beta, 0.0, 1.0)
    mu = torch.clamp(mu, min=0.0)
    return pack_pooled_params(beta, mu)


def split_aff_params(x: torch.Tensor, k_sources: int, d_ctx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s0 = 2
    s1 = s0 + k_sources
    s2 = s1 + k_sources * d_ctx
    beta = x[:s0]
    mu = x[s0:s1]
    a = x[s1:s2].reshape(k_sources, d_ctx)
    a_mat = x[s2:].reshape(k_sources, d_ctx, d_ctx)
    return beta, mu, a, a_mat


def pack_aff_params(beta: torch.Tensor, mu: torch.Tensor, a: torch.Tensor, a_mat: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            beta.reshape(-1),
            mu.reshape(-1),
            a.reshape(-1),
            a_mat.reshape(-1),
        ],
        dim=0,
    )


def project_aff_params(x: torch.Tensor, k_sources: int, d_ctx: int) -> torch.Tensor:
    beta, mu, a, a_mat = split_aff_params(x, k_sources, d_ctx)

    beta = torch.clamp(beta, 0.0, 1.0)
    mu = torch.clamp(mu, min=0.0)

    # Enforce affine coupling constraints: sum_k a_k = 0 and sum_k A_k = 0.
    a = a - a.mean(dim=0, keepdim=True)

    # Keep A_k symmetric, then center so sum_k A_k = 0.
    a_mat = 0.5 * (a_mat + a_mat.transpose(1, 2))
    a_mat = a_mat - a_mat.mean(dim=0, keepdim=True)

    return pack_aff_params(beta, mu, a, a_mat)


def policy_prob_threshold(
    contexts: torch.Tensor,
    beta: torch.Tensor,
    feature_idx: int,
    threshold: float,
) -> torch.Tensor:
    mask = contexts[..., feature_idx] <= threshold
    return torch.where(mask, beta[0], beta[1])


def expected_cost(
    contexts: torch.Tensor,
    treatment: torch.Tensor,
    control: torch.Tensor,
    beta: torch.Tensor,
    feature_idx: int,
    threshold: float,
) -> torch.Tensor:
    p_treat = policy_prob_threshold(contexts, beta, feature_idx, threshold)
    return p_treat * treatment + (1.0 - p_treat) * control


def weighted_average(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights / torch.clamp(weights.sum(), min=1e-12)
    return (values * w).sum()


def mlmc_level_probs(rate: float, truncation: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if not (0.0 < rate < 1.0):
        raise ValueError(f"MLMC rate must be in (0,1), got {rate}")
    levels = torch.arange(truncation + 1, dtype=dtype, device=device)
    raw = rate * torch.pow(1.0 - rate, levels)
    return raw / raw.sum()


def draw_levels(level_probs: torch.Tensor, n: int) -> torch.Tensor:
    return torch.multinomial(level_probs, num_samples=n, replacement=True)


def validate_prob_vector(v: Sequence[float], name: str, tol: float = 1e-8) -> None:
    if len(v) == 0:
        raise ValueError(f"{name} must not be empty")
    if any(x < 0 for x in v):
        raise ValueError(f"{name} must be nonnegative")
    s = float(sum(v))
    if abs(s - 1.0) > tol:
        raise ValueError(f"{name} must sum to 1.0, got {s}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_estimators(raw: str) -> List[str]:
    vals = [v.strip().lower() for v in parse_str_list(raw)]
    if len(vals) == 0:
        vals = ["saa", "mlmc"]
    for v in vals:
        if v not in {"saa", "mlmc"}:
            raise ValueError(f"Unknown estimator: {v}")
    return vals


def _resolve_policy_feature_idx(data: Any, policy_feature: str) -> int:
    if policy_feature == "":
        return 0
    if policy_feature.isdigit():
        idx = int(policy_feature)
        if idx < 0 or idx >= data.d_context:
            raise ValueError(f"policy_feature index out of range: {idx}")
        return idx
    if policy_feature not in data.context_cols:
        raise ValueError(
            f"policy_feature={policy_feature} not in context columns {data.context_cols}. "
            "Pass either a valid name or an integer index."
        )
    return data.context_cols.index(policy_feature)


def _make_aff_initial(
    beta_init: List[float],
    mu_init_raw: List[float],
    a_init: float,
    a_mat_init: float,
    k_sources: int,
    d_ctx: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    beta = torch.as_tensor(beta_init, dtype=dtype)

    if len(mu_init_raw) == 1:
        mu = torch.full((k_sources,), float(mu_init_raw[0]), dtype=dtype)
    elif len(mu_init_raw) == k_sources:
        mu = torch.as_tensor(mu_init_raw, dtype=dtype)
    else:
        raise ValueError(
            f"aff_mu_init must have length 1 or K={k_sources}, got {len(mu_init_raw)}"
        )

    a = torch.full((k_sources, d_ctx), float(a_init), dtype=dtype)
    a_mat = torch.full((k_sources, d_ctx, d_ctx), float(a_mat_init), dtype=dtype)

    x0 = torch.cat([beta, mu, a.reshape(-1), a_mat.reshape(-1)], dim=0)
    return project_aff_params(x0, k_sources, d_ctx)


def _make_pooled_initial(beta_init: List[float], mu_init: float, dtype: torch.dtype) -> torch.Tensor:
    x0 = torch.as_tensor([beta_init[0], beta_init[1], mu_init], dtype=dtype)
    return project_pooled_params(x0)


def _train(
    x0: torch.Tensor,
    num_iters: int,
    learning_rate: float,
    optimizer_name: str,
    estimator_fn,
    project_fn,
    n1: int,
) -> Dict[str, np.ndarray]:
    x = x0.detach().clone().requires_grad_(True)

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam([x], lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD([x], lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    traj = [x.detach().cpu().numpy().copy()]
    objective_history: List[float] = []
    sample_cost_history: List[float] = []
    cumulative_cost = 0.0

    for _ in range(num_iters):
        optimizer.zero_grad()

        est = estimator_fn(x, n1)
        x.grad = est.grad
        optimizer.step()

        with torch.no_grad():
            x.copy_(project_fn(x))

        cumulative_cost += float(est.sample_cost)
        objective_history.append(float(est.objective_estimate))
        sample_cost_history.append(float(cumulative_cost))
        traj.append(x.detach().cpu().numpy().copy())

    return {
        "x": np.asarray(traj, dtype=float),
        "objective": np.asarray(objective_history, dtype=float),
        "sample_cost": np.asarray(sample_cost_history, dtype=float),
    }


def _record_aff(train_out: Dict[str, np.ndarray], meta: Dict, k_sources: int, d_ctx: int) -> Dict:
    x = train_out["x"]
    beta = x[:, :2]
    mu_k = x[:, 2 : 2 + k_sources]

    return {
        **meta,
        "beta": beta,
        "mu_k": mu_k,
        "objective": train_out["objective"],
        "sample_cost": train_out["sample_cost"],
        "x": x,
    }


def _record_pooled(train_out: Dict[str, np.ndarray], meta: Dict) -> Dict:
    x = train_out["x"]
    beta = x[:, :2]
    mu = x[:, 2]

    return {
        **meta,
        "beta": beta,
        "mu": mu,
        "objective": train_out["objective"],
        "sample_cost": train_out["sample_cost"],
        "x": x,
    }


def _save_run_npz(path: str, rec: Dict) -> None:
    arrs = {k: v for k, v in rec.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in rec.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrs, __meta__=json.dumps(meta))
