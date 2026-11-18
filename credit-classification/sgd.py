"""The subgradient method of Section 4 for the robust methods, stochastic gradient descent for ORACLE, and the
out-of-sample cross-entropy. Every function trains or evaluates M networks (one per simulation) in lockstep."""
from __future__ import annotations

import torch

from methods import BatchedMLP, Problem, bce, init_net, mlp

LR_NET, LR_MULT, LR_RULE = 0.05, 0.05, 0.0125   # network and b and cell coefficients d; coefficients a, A
ORACLE_BATCH, ORACLE_CHECKPOINTS = 128, [2_500] + list(range(5_000, 75_001, 5_000))


def networks(init_seeds: list, device) -> BatchedMLP:
    return BatchedMLP([init_net(s) for s in init_seeds]).to(device)


def train_robust(P: Problem, net: BatchedMLP, steps: int, log=print) -> BatchedMLP:
    """Projected subgradient method with constant step sizes; the projection enforces the zero-sum constraints."""
    params = list(net.parameters())
    for t in range(1, steps + 1):
        J = P.objective(net)
        for v in params + list(P.duals.values()):
            v.grad = None
        J.sum().backward()
        with torch.no_grad():
            for p in params:
                p -= LR_NET * p.grad
            for k, v in P.duals.items():
                v -= (LR_RULE if k in ("a", "A") else LR_MULT) * v.grad
            P.project()
        if t % 5_000 == 0:
            log(f"update {t}: mean objective {J.mean().item():.4f}")
    return net


@torch.no_grad()
def test_ce(net: BatchedMLP, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy (M,) in float64 on the test records X (M, n, 59), y (M, n) of each simulation."""
    return bce(mlp([W.double() for W in net.W], [b.double() for b in net.b], X.double()), y).mean(1)


def train_oracle(net: BatchedMLP, X: torch.Tensor, y: torch.Tensor, batch_seeds: list, Xt, yt, log=print) -> tuple:
    """Empirical risk minimization on the hidden records X (M, N, 59), y (M, N) with minibatches of 128 and step size
    0.05. Returns the lowest test cross-entropy over the checkpoints (hindsight) and its update, per simulation."""
    gens = [torch.Generator().manual_seed(int(s)) for s in batch_seeds]
    M, N = y.shape
    best_ce = torch.full((M,), float("inf"), dtype=torch.float64, device=X.device)
    best_step = torch.zeros(M, dtype=torch.long, device=X.device)
    for t in range(1, ORACLE_CHECKPOINTS[-1] + 1):
        if (t - 1) % 2_500 == 0:   # minibatch indices are drawn in blocks of 2,500 updates
            block = torch.stack([torch.randint(0, N, (2_500, ORACLE_BATCH), generator=g) for g in gens], 1).to(X.device)
        rows = block[(t - 1) % 2_500]
        Xb = X.gather(1, rows[..., None].expand(-1, -1, X.shape[-1]))
        loss = bce(net(Xb), y.gather(1, rows)).mean(1)
        for p in net.parameters():
            p.grad = None
        loss.sum().backward()
        with torch.no_grad():
            for p in net.parameters():
                p -= LR_NET * p.grad
        if t in ORACLE_CHECKPOINTS:
            ce = test_ce(net, Xt, yt)
            better = ce < best_ce
            best_ce, best_step = torch.where(better, ce, best_ce), torch.where(better, t, best_step)
            log(f"update {t}: mean test cross-entropy {ce.mean().item():.4f}")
    return best_ce, best_step
