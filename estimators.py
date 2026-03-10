import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

from simulators import (
    ExcelBanditData,
    sample_inner_uniform,
    sample_outer_by_source,
    sample_outer_pooled,
)
from utils import (
    draw_levels,
    mlmc_level_probs,
    policy_prob_threshold,
    split_aff_params,
    split_pooled_params,
)


@dataclass
class GradEstimate:
    grad: torch.Tensor
    sample_cost: float
    objective_estimate: float


Estimator = Callable[[torch.Tensor, int], GradEstimate]


@dataclass
class EstimatorConfig:
    estimator: str
    n1: int
    saa_inner: int
    mlmc_rate: float
    mlmc_truncation: int
    gamma: float
    policy_feature_idx: int
    policy_threshold: float
    include_linear_mu_eps_term: bool = True


def _aff_phi(
    beta: torch.Tensor,
    mu_k: torch.Tensor,
    a_k: torch.Tensor,
    a_mat_k: torch.Tensor,
    t_k: float,
    eps_k: float,
    xi_prime: torch.Tensor,
    u_ctx: torch.Tensor,
    u_treat: torch.Tensor,
    u_ctrl: torch.Tensor,
    gamma: float,
    policy_feature_idx: int,
    policy_threshold: float,
    include_linear_mu_eps_term: bool,
) -> torch.Tensor:
    p_treat = policy_prob_threshold(u_ctx, beta, policy_feature_idx, policy_threshold)
    ell = p_treat * u_treat + (1.0 - p_treat) * u_ctrl

    linear = u_ctx @ a_k
    quad = torch.einsum("nd,dd,nd->n", u_ctx, a_mat_k, u_ctx)
    dist2 = ((u_ctx - xi_prime.unsqueeze(0)) ** 2).sum(dim=1)

    f = t_k * ell + linear + quad - mu_k * dist2
    if include_linear_mu_eps_term:
        f = f + mu_k * eps_k

    return (torch.logsumexp(gamma * f, dim=0) - math.log(u_ctx.shape[0])) / gamma


def _pooled_phi(
    beta: torch.Tensor,
    mu: torch.Tensor,
    eps_pool: float,
    xi_prime: torch.Tensor,
    u_ctx: torch.Tensor,
    u_treat: torch.Tensor,
    u_ctrl: torch.Tensor,
    gamma: float,
    policy_feature_idx: int,
    policy_threshold: float,
    include_linear_mu_eps_term: bool,
) -> torch.Tensor:
    p_treat = policy_prob_threshold(u_ctx, beta, policy_feature_idx, policy_threshold)
    ell = p_treat * u_treat + (1.0 - p_treat) * u_ctrl

    dist2 = ((u_ctx - xi_prime.unsqueeze(0)) ** 2).sum(dim=1)
    f = ell - mu * dist2
    if include_linear_mu_eps_term:
        f = f + mu * eps_pool

    return (torch.logsumexp(gamma * f, dim=0) - math.log(u_ctx.shape[0])) / gamma


def _aff_objective_saa(
    x: torch.Tensor,
    data: ExcelBanditData,
    epsilons: Sequence[float],
    t_weights: Sequence[float],
    cfg: EstimatorConfig,
    outer_n: int,
) -> tuple[torch.Tensor, float]:
    beta, mu, a, a_mat = split_aff_params(x, data.n_sources, data.d_context)

    objective = torch.zeros((), dtype=x.dtype, device=x.device)
    sample_cost = 0.0

    for k in range(data.n_sources):
        xi_batch = sample_outer_by_source(data, k, outer_n, split="train").to(x.device)

        terms = []
        for i in range(outer_n):
            u_ctx, u_treat, u_ctrl = sample_inner_uniform(data, cfg.saa_inner)
            u_ctx = u_ctx.to(x.device)
            u_treat = u_treat.to(x.device)
            u_ctrl = u_ctrl.to(x.device)
            terms.append(
                _aff_phi(
                    beta=beta,
                    mu_k=mu[k],
                    a_k=a[k],
                    a_mat_k=a_mat[k],
                    t_k=float(t_weights[k]),
                    eps_k=float(epsilons[k]),
                    xi_prime=xi_batch[i],
                    u_ctx=u_ctx,
                    u_treat=u_treat,
                    u_ctrl=u_ctrl,
                    gamma=cfg.gamma,
                    policy_feature_idx=cfg.policy_feature_idx,
                    policy_threshold=cfg.policy_threshold,
                    include_linear_mu_eps_term=cfg.include_linear_mu_eps_term,
                )
            )

        source_mc = torch.stack(terms).mean()
        objective = objective + mu[k] * (float(epsilons[k]) ** 2) + source_mc
        sample_cost += float(outer_n * cfg.saa_inner)

    return objective, sample_cost


def _aff_objective_mlmc(
    x: torch.Tensor,
    data: ExcelBanditData,
    epsilons: Sequence[float],
    t_weights: Sequence[float],
    cfg: EstimatorConfig,
    outer_n: int,
) -> tuple[torch.Tensor, float]:
    beta, mu, a, a_mat = split_aff_params(x, data.n_sources, data.d_context)

    objective = torch.zeros((), dtype=x.dtype, device=x.device)
    sample_cost = 0.0

    lvl_probs = mlmc_level_probs(
        rate=cfg.mlmc_rate,
        truncation=cfg.mlmc_truncation,
        dtype=x.dtype,
        device=x.device,
    )

    for k in range(data.n_sources):
        xi_batch = sample_outer_by_source(data, k, outer_n, split="train").to(x.device)
        levels = draw_levels(lvl_probs, outer_n)

        unbiased_terms = []
        for i in range(outer_n):
            lvl = int(levels[i].item())
            q_lvl = lvl_probs[lvl]
            n_inner = 2 ** lvl

            u_ctx, u_treat, u_ctrl = sample_inner_uniform(data, n_inner)
            u_ctx = u_ctx.to(x.device)
            u_treat = u_treat.to(x.device)
            u_ctrl = u_ctrl.to(x.device)

            fine = _aff_phi(
                beta=beta,
                mu_k=mu[k],
                a_k=a[k],
                a_mat_k=a_mat[k],
                t_k=float(t_weights[k]),
                eps_k=float(epsilons[k]),
                xi_prime=xi_batch[i],
                u_ctx=u_ctx,
                u_treat=u_treat,
                u_ctrl=u_ctrl,
                gamma=cfg.gamma,
                policy_feature_idx=cfg.policy_feature_idx,
                policy_threshold=cfg.policy_threshold,
                include_linear_mu_eps_term=cfg.include_linear_mu_eps_term,
            )

            if lvl == 0:
                unbiased = fine / q_lvl
            else:
                even_ctx = u_ctx[0::2]
                odd_ctx = u_ctx[1::2]
                even_treat = u_treat[0::2]
                odd_treat = u_treat[1::2]
                even_ctrl = u_ctrl[0::2]
                odd_ctrl = u_ctrl[1::2]

                coarse_even = _aff_phi(
                    beta=beta,
                    mu_k=mu[k],
                    a_k=a[k],
                    a_mat_k=a_mat[k],
                    t_k=float(t_weights[k]),
                    eps_k=float(epsilons[k]),
                    xi_prime=xi_batch[i],
                    u_ctx=even_ctx,
                    u_treat=even_treat,
                    u_ctrl=even_ctrl,
                    gamma=cfg.gamma,
                    policy_feature_idx=cfg.policy_feature_idx,
                    policy_threshold=cfg.policy_threshold,
                    include_linear_mu_eps_term=cfg.include_linear_mu_eps_term,
                )
                coarse_odd = _aff_phi(
                    beta=beta,
                    mu_k=mu[k],
                    a_k=a[k],
                    a_mat_k=a_mat[k],
                    t_k=float(t_weights[k]),
                    eps_k=float(epsilons[k]),
                    xi_prime=xi_batch[i],
                    u_ctx=odd_ctx,
                    u_treat=odd_treat,
                    u_ctrl=odd_ctrl,
                    gamma=cfg.gamma,
                    policy_feature_idx=cfg.policy_feature_idx,
                    policy_threshold=cfg.policy_threshold,
                    include_linear_mu_eps_term=cfg.include_linear_mu_eps_term,
                )
                unbiased = (fine - 0.5 * (coarse_even + coarse_odd)) / q_lvl

            unbiased_terms.append(unbiased)
            sample_cost += float(n_inner)

        source_mlmc = torch.stack(unbiased_terms).mean()
        objective = objective + mu[k] * (float(epsilons[k]) ** 2) + source_mlmc

    return objective, sample_cost


def _pooled_objective_saa(
    x: torch.Tensor,
    data: ExcelBanditData,
    epsilon_pool: float,
    cfg: EstimatorConfig,
    outer_n: int,
) -> tuple[torch.Tensor, float]:
    beta, mu = split_pooled_params(x)

    xi_batch = sample_outer_pooled(data, outer_n, split="train").to(x.device)

    terms = []
    for i in range(outer_n):
        u_ctx, u_treat, u_ctrl = sample_inner_uniform(data, cfg.saa_inner)
        u_ctx = u_ctx.to(x.device)
        u_treat = u_treat.to(x.device)
        u_ctrl = u_ctrl.to(x.device)

        terms.append(
            _pooled_phi(
                beta=beta,
                mu=mu[0],
                eps_pool=epsilon_pool,
                xi_prime=xi_batch[i],
                u_ctx=u_ctx,
                u_treat=u_treat,
                u_ctrl=u_ctrl,
                gamma=cfg.gamma,
                policy_feature_idx=cfg.policy_feature_idx,
                policy_threshold=cfg.policy_threshold,
                include_linear_mu_eps_term=cfg.include_linear_mu_eps_term,
            )
        )

    objective = mu[0] * (float(epsilon_pool) ** 2) + torch.stack(terms).mean()
    sample_cost = float(outer_n * cfg.saa_inner)
    return objective, sample_cost


def _pooled_objective_mlmc(
    x: torch.Tensor,
    data: ExcelBanditData,
    epsilon_pool: float,
    cfg: EstimatorConfig,
    outer_n: int,
) -> tuple[torch.Tensor, float]:
    beta, mu = split_pooled_params(x)

    xi_batch = sample_outer_pooled(data, outer_n, split="train").to(x.device)
    lvl_probs = mlmc_level_probs(
        rate=cfg.mlmc_rate,
        truncation=cfg.mlmc_truncation,
        dtype=x.dtype,
        device=x.device,
    )
    levels = draw_levels(lvl_probs, outer_n)

    unbiased_terms = []
    sample_cost = 0.0
    for i in range(outer_n):
        lvl = int(levels[i].item())
        q_lvl = lvl_probs[lvl]
        n_inner = 2 ** lvl

        u_ctx, u_treat, u_ctrl = sample_inner_uniform(data, n_inner)
        u_ctx = u_ctx.to(x.device)
        u_treat = u_treat.to(x.device)
        u_ctrl = u_ctrl.to(x.device)

        fine = _pooled_phi(
            beta=beta,
            mu=mu[0],
            eps_pool=epsilon_pool,
            xi_prime=xi_batch[i],
            u_ctx=u_ctx,
            u_treat=u_treat,
            u_ctrl=u_ctrl,
            gamma=cfg.gamma,
            policy_feature_idx=cfg.policy_feature_idx,
            policy_threshold=cfg.policy_threshold,
            include_linear_mu_eps_term=cfg.include_linear_mu_eps_term,
        )

        if lvl == 0:
            unbiased = fine / q_lvl
        else:
            even_ctx = u_ctx[0::2]
            odd_ctx = u_ctx[1::2]
            even_treat = u_treat[0::2]
            odd_treat = u_treat[1::2]
            even_ctrl = u_ctrl[0::2]
            odd_ctrl = u_ctrl[1::2]

            coarse_even = _pooled_phi(
                beta=beta,
                mu=mu[0],
                eps_pool=epsilon_pool,
                xi_prime=xi_batch[i],
                u_ctx=even_ctx,
                u_treat=even_treat,
                u_ctrl=even_ctrl,
                gamma=cfg.gamma,
                policy_feature_idx=cfg.policy_feature_idx,
                policy_threshold=cfg.policy_threshold,
                include_linear_mu_eps_term=cfg.include_linear_mu_eps_term,
            )
            coarse_odd = _pooled_phi(
                beta=beta,
                mu=mu[0],
                eps_pool=epsilon_pool,
                xi_prime=xi_batch[i],
                u_ctx=odd_ctx,
                u_treat=odd_treat,
                u_ctrl=odd_ctrl,
                gamma=cfg.gamma,
                policy_feature_idx=cfg.policy_feature_idx,
                policy_threshold=cfg.policy_threshold,
                include_linear_mu_eps_term=cfg.include_linear_mu_eps_term,
            )
            unbiased = (fine - 0.5 * (coarse_even + coarse_odd)) / q_lvl

        unbiased_terms.append(unbiased)
        sample_cost += float(n_inner)

    objective = mu[0] * (float(epsilon_pool) ** 2) + torch.stack(unbiased_terms).mean()
    return objective, sample_cost


def make_aff_estimator(
    data: ExcelBanditData,
    epsilons: Sequence[float],
    t_weights: Sequence[float],
    cfg: EstimatorConfig,
) -> Estimator:
    if cfg.estimator not in {"saa", "mlmc"}:
        raise ValueError(f"Unknown estimator: {cfg.estimator}")

    def _estimate(x: torch.Tensor, outer_n: Optional[int] = None) -> GradEstimate:
        n = int(cfg.n1 if outer_n is None else outer_n)
        if cfg.estimator == "saa":
            obj, sample_cost = _aff_objective_saa(x, data, epsilons, t_weights, cfg, n)
        else:
            obj, sample_cost = _aff_objective_mlmc(x, data, epsilons, t_weights, cfg, n)

        grad = torch.autograd.grad(obj, x, retain_graph=False, create_graph=False)[0]
        return GradEstimate(
            grad=grad.detach(),
            sample_cost=float(sample_cost),
            objective_estimate=float(obj.detach().cpu().item()),
        )

    return _estimate


def make_pooled_estimator(
    data: ExcelBanditData,
    epsilon_pool: float,
    cfg: EstimatorConfig,
) -> Estimator:
    if cfg.estimator not in {"saa", "mlmc"}:
        raise ValueError(f"Unknown estimator: {cfg.estimator}")

    def _estimate(x: torch.Tensor, outer_n: Optional[int] = None) -> GradEstimate:
        n = int(cfg.n1 if outer_n is None else outer_n)
        if cfg.estimator == "saa":
            obj, sample_cost = _pooled_objective_saa(x, data, epsilon_pool, cfg, n)
        else:
            obj, sample_cost = _pooled_objective_mlmc(x, data, epsilon_pool, cfg, n)

        grad = torch.autograd.grad(obj, x, retain_graph=False, create_graph=False)[0]
        return GradEstimate(
            grad=grad.detach(),
            sample_cost=float(sample_cost),
            objective_estimate=float(obj.detach().cpu().item()),
        )

    return _estimate
