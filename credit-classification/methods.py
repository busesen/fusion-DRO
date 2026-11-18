"""Objectives of the methods of Section 5.3, evaluated for M simulations at once (one network per simulation).

Elementary ambiguity sets: B_1 (moments, ||E_Q[phi] - m_hat||_2 <= r_M) and B_2 (histogram, TV(p_Q, p_hat) <= r_TV).
  affine     B_1 and B_2, t_k = 1/2, lambda_k = t_k l + a_k^T s + <A_k, chi chi^T> + d_{k, cell}
  static     the same with A_k = 0
  minkowski  B_1 and B_2, t_k = 1/2, lambda_k = t_k l (no decision rule coefficients)
  moment     B_1 alone, lambda = l
  tv         B_2 alone, lambda = l
Here chi = (x, y) stacks the 59 network inputs and the label, s = (chi, cell indicators), and every decision rule
coefficient sums to zero over k. The worst case over B_1 is replaced by its dual
    b^T m_hat + r_M ||b||_2 + max_xi [lambda_1(xi) - b^T phi(xi)],
whose multiplier b is updated with the network. The worst case over B_2 places the probability of every cell on a
maximizer of lambda_2 in that cell and moves probability r_TV from the cells with the smallest maxima to the cell with
the largest one. The maximizations over Xi use multi-start projected gradient ascent with rounding (domain.py); the
objective and its gradient (Danskin) are then evaluated at the chosen points.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import domain as DM
from credit_data import IU, N_CELLS, N_IN, N_NUM, N_PHI

ARMS = {"affine": ("M", "TV"), "static": ("M", "TV"), "minkowski": ("M", "TV"), "moment": ("M",), "tv": ("TV",)}
N_RULE = N_IN + 1                     # chi = (x, y)
SEARCH = {"starts": 8, "steps": 32, "eta_z": 0.05, "eta_u": 0.1, "perturb_sigma": 0.15}
WIDTHS, LOGIT_CAP = (256, 128, 64), 8.0


# ------------------------------------------------------------------------------------------------ network
class Net(nn.Module):
    """59 -> 256 -> 128 -> 64 -> 1 with SiLU activations; the output logit g is capped as 8 tanh(g / 8)."""

    def __init__(self):
        super().__init__()
        layers, d = [], N_IN
        for w in WIDTHS:
            layers += [nn.Linear(d, w), nn.SiLU()]
            d = w
        self.body = nn.Sequential(*layers, nn.Linear(d, 1))


def init_net(seed: int) -> Net:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Net()


class BatchedMLP(nn.Module):
    """M independent networks evaluated with batched matrix products (layer l of network m: x @ W[l][m] + b[l][m])."""

    def __init__(self, nets: list):
        super().__init__()
        lins = [[mod for mod in n.body if isinstance(mod, nn.Linear)] for n in nets]
        self.W = nn.ParameterList([nn.Parameter(torch.stack([ls[i].weight.detach().T.clone() for ls in lins]))
                                   for i in range(len(lins[0]))])
        self.b = nn.ParameterList([nn.Parameter(torch.stack([ls[i].bias.detach().clone() for ls in lins]))
                                   for i in range(len(lins[0]))])

    def forward(self, X):
        return mlp(self.W, self.b, X)


def mlp(Ws, bs, X):
    """logits (M, n) of the M networks with weights Ws, biases bs at inputs X (M, n, 59)"""
    h = X
    for i, (W, b) in enumerate(zip(Ws, bs)):
        h = torch.matmul(h, W) + b[:, None, :]
        if i < len(Ws) - 1:
            h = F.silu(h)
    return LOGIT_CAP * torch.tanh(h.squeeze(-1) / LOGIT_CAP)


def bce(f, y):
    """cross-entropy of logit f and label y: softplus(f) - y f"""
    return F.softplus(f) - y.to(f.dtype) * f


# ------------------------------------------------------------------------------------------------ helpers
def safe_norm(b):
    """row norms with subgradient 0 at 0"""
    sq = (b * b).sum(-1)
    nz = sq > 0
    return torch.where(nz, torch.sqrt(torch.where(nz, sq, torch.ones_like(sq))), torch.zeros_like(sq))


def phi_dot(b, z, y):
    """b^T phi(z, y): b (M, 134), z (M, k, 14), y (M, k) -> (M, k)"""
    i0, i1 = DM._c("iu0", z, IU[0], torch.long), DM._c("iu1", z, IU[1], torch.long)
    n_v = N_NUM + len(IU[0])
    base = (z * b[:, None, :N_NUM]).sum(-1) + (z[..., i0] * z[..., i1] * b[:, None, N_NUM:n_v]).sum(-1)
    return base + y.to(z.dtype) * ((z * b[:, None, n_v:N_PHI - 1]).sum(-1) + b[:, None, -1])


def rule(a, A, chi):
    """a^T chi + <A, chi chi^T>: a (M, 60), A (M, 60, 60) or None, chi (M, k, 60) -> (M, k)"""
    v = (chi * a[:, None, :]).sum(-1)
    return v if A is None else v + (torch.matmul(chi, A) * chi).sum(-1)


def tv_worst_case(v, p, r):
    """Maximizer q (M, 432) of sum_c q_c v_c over histograms with TV(q, p) <= r: probability r moves from the cells
    with the smallest values (each giving up at most p_c) to the cell with the largest value."""
    with torch.no_grad():
        k = v.argmax(1)
        order = torch.argsort(v.scatter(1, k[:, None], float("inf")), dim=1, stable=True)
        ps = p.gather(1, order).clone()
        ps[:, -1] = 0.0                                  # the receiving cell sorts last
        budget = torch.minimum(1.0 - p.gather(1, k[:, None]).squeeze(1), r)
        take = torch.clamp(torch.minimum(budget[:, None] - (torch.cumsum(ps, 1) - ps), ps), min=0.0)
        return p.clone().scatter_add(1, order, -take).scatter_add(1, k[:, None], take.sum(1, keepdim=True))


# ------------------------------------------------------------------------------------------------ the objective
class Problem:
    """Decision rule coefficients, multipliers and stochastic objective of one method for M simulations.
    rel: released statistics stacked over the simulations, m_hat (M, 134), r_M (M,), p_hat (M, 432), r_TV (M,)."""

    def __init__(self, method: str, rel: dict, seed: int, cuda_graph: bool = True):
        self.arms = ARMS[method]
        self.K, self.t = len(self.arms), 1.0 / len(self.arms)
        self.rel = rel
        M, dev, dt = rel["p_hat"].shape[0], rel["p_hat"].device, rel["p_hat"].dtype
        z = lambda *s: torch.zeros(*s, device=dev, dtype=dt, requires_grad=True)
        self.duals = {}
        if method in ("affine", "static"):
            self.duals["a"] = z(M, self.K, N_RULE)
            self.duals["d"] = z(M, self.K, N_CELLS)      # coefficients of the cell indicators
            if method == "affine":
                self.duals["A"] = z(M, self.K, N_RULE, N_RULE)
        if "M" in self.arms:
            self.duals["b"] = z(M, N_PHI)
        self.M, self.dev, self.dtype = M, dev, dt
        self.gen = torch.Generator(device=dev).manual_seed(int(seed))
        self.use_graph = cuda_graph and dev.type == "cuda"
        self.prev, self.graphs = {}, {}

    def coef(self, k):
        if "a" not in self.duals:
            return None, None
        i = self.arms.index(k)
        return self.duals["a"][:, i], (self.duals["A"][:, i] if "A" in self.duals else None)

    @torch.no_grad()
    def project(self):
        """zero-sum constraints: subtract the average over k from every decision rule coefficient"""
        for key in ("a", "A", "d"):
            if key in self.duals:
                self.duals[key] -= self.duals[key].mean(1, keepdim=True)

    def _value(self, segs, X, y, cell, net, detach=True):
        """integrands of the inner maximizations at points X (M, G, R, 59) of groups G with labels y and cells cell
        (M, G); segs = [(arm, first group, last group, b)] concatenates the arms along the group axis"""
        M_, G_, R_ = X.shape[:3]
        dd = (lambda v: v.detach()) if detach else (lambda v: v)
        yR = y[:, :, None].expand(M_, G_, R_)
        lam = self.t * bce(net(X.reshape(M_, G_ * R_, -1)), yR.reshape(M_, -1)).reshape(M_, G_, R_)
        out = []
        for k, g0, g1, b in segs:
            Xk, yk, v = X[:, g0:g1], yR[:, g0:g1], lam[:, g0:g1]
            n_ = (g1 - g0) * R_
            a, A = self.coef(k)
            if a is not None:
                chi = torch.cat([Xk, yk.to(X.dtype)[..., None]], -1).reshape(M_, n_, -1)
                v = v + rule(dd(a), dd(A) if A is not None else None, chi).reshape(M_, g1 - g0, R_)
                v = v + dd(self.duals["d"][:, self.arms.index(k)]).gather(1, cell[:, g0:g1])[:, :, None]
            if k == "M":
                v = v - phi_dot(b, Xk[..., :N_NUM].reshape(M_, n_, -1), yk.reshape(M_, n_)).reshape(M_, g1 - g0, R_)
            out.append(v)
        return torch.cat(out, 1)

    def _search(self, segs, X0, y, Rb, Fb, Sx, cell, net):
        """projected gradient ascent from the starts X0 (M, G, S, 59) on the relaxation, rounding into the case; the
        better of each start and its rounded end point; then the best start of every group"""
        o, sh = SEARCH, X0.shape[:3]
        RbS, FbS, SxS = Rb[:, :, None].expand(sh), Fb[:, :, None].expand(sh), Sx[:, :, None, :].expand(*sh, 3)
        X = X0.detach().clone()
        for _ in range(o["steps"]):
            X.requires_grad_(True)
            g, = torch.autograd.grad(self._value(segs, X, y, cell, net).sum(), X)
            with torch.no_grad():
                gu = g[..., N_NUM:]
                gu = gu / gu.abs().amax(-1, keepdim=True).clamp_min(1e-12)
                X = DM.project_relaxed(torch.cat([X[..., :N_NUM] + o["eta_z"] * g[..., :N_NUM].sign(),
                                                  X[..., N_NUM:] + o["eta_u"] * gu], -1), RbS, SxS)
        with torch.no_grad():
            Xr = DM.round_valid(X, RbS, FbS, SxS)
            h0, h1 = self._value(segs, X0, y, cell, net), self._value(segs, Xr, y, cell, net)
            Xb, hb = torch.where((h1 > h0)[..., None], Xr, X0), torch.maximum(h0, h1)
            j = hb.argmax(-1)
            return Xb.gather(2, j[..., None, None].expand(-1, -1, 1, Xb.shape[-1])).squeeze(2), hb.max(-1).values

    def _search_graphed(self, segs, *inputs, net):
        """_search replayed from a CUDA graph captured once (inputs are copied into static buffers; the network
        weights and multipliers are read in place)"""
        if "g" not in self.graphs:
            stat = [T.clone() for T in inputs]
            sb = [None if s[3] is None else s[3].clone() for s in segs]
            ssegs = [(k, g0, g1, b) for (k, g0, g1, _), b in zip(segs, sb)]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._search(ssegs, *stat, net)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._search(ssegs, *stat, net)
            self.graphs["g"] = (graph, stat, sb, out)
        graph, stat, sb, out = self.graphs["g"]
        for dst, src in zip(stat, inputs):
            dst.copy_(src)
        for dst, s in zip(sb, segs):
            if dst is not None:
                dst.copy_(s[3])
        graph.replay()
        return out[0].clone(), out[1].clone()

    def objective(self, net):
        """stochastic objective J (M,) whose gradient is the subgradient of the method's dual objective"""
        M_, dev, S = self.M, self.dev, SEARCH["starts"]
        n_pert = S // 2 - 1
        n_rnd = S - 1 - n_pert
        b = self.duals["b"].detach() if "b" in self.duals else None
        segs, st, gy, gR, gF, gS, gc, g0 = [], [], [], [], [], [], [], 0
        with torch.no_grad():
            for k in self.arms:   # cases: the 432 cells, or the 2 labels for the moment set without cell coefficients
                cases = DM.cell_cases if ("d" in self.duals or k == "TV") else DM.label_cases
                yC, RbC, FbC, SxC, cellC = cases(dev)
                C_ = len(yC)
                Rb, Fb = RbC[None, :, None].expand(M_, C_, 1), FbC[None, :, None].expand(M_, C_, 1)
                Sx = SxC[None, :, None, :].expand(M_, C_, 1, 3)
                rnd = DM.random_valid((M_, C_, n_rnd + 1), Rb, Fb, Sx, self.gen, dev, self.dtype)
                prev = self.prev.get(k, rnd[:, :, :1])       # the case's maximizer of the previous update
                pert = DM.perturb_valid(prev.expand(-1, -1, n_pert, -1), Rb, Fb, Sx, SEARCH["perturb_sigma"], self.gen)
                st.append(torch.cat([prev, pert, rnd[:, :, 1:]], 2))
                segs.append((k, g0, g0 + C_, b if k == "M" else None))
                for lst, T in ((gy, yC), (gR, RbC), (gF, FbC), (gc, cellC)):
                    lst.append(T[None].expand(M_, -1))
                gS.append(SxC[None].expand(M_, -1, -1))
                g0 += C_
        inputs = [torch.cat(T, 1) for T in (st, gy, gR, gF, gS, gc)]
        torch.backends.cuda.matmul.allow_tf32 = True       # TF32 in the search only
        Xb, hb = (self._search_graphed(segs, *inputs, net=net) if self.use_graph else self._search(segs, *inputs, net))
        torch.backends.cuda.matmul.allow_tf32 = False
        gy, gc = inputs[1], inputs[5]
        J = torch.zeros(M_, device=dev, dtype=self.dtype)
        for k, s0, s1, _ in segs:
            with torch.no_grad():
                Xk, yk, ck = Xb[:, s0:s1], gy[:, s0:s1], gc[:, s0:s1]
                self.prev[k] = Xk[:, :, None, :].clone()
                if k == "M":                                  # the maximizer over all cases
                    j = hb[:, s0:s1].argmax(-1, keepdim=True)
                    Xk = Xk.gather(1, j[..., None].expand(-1, -1, Xk.shape[-1]))
                    yk, ck = yk.gather(1, j), ck.gather(1, j)
            lam = self.t * bce(net(Xk), yk)
            a, A = self.coef(k)
            if a is not None:
                lam = lam + rule(a, A, torch.cat([Xk, yk.to(self.dtype)[..., None]], -1))
                lam = lam + self.duals["d"][:, self.arms.index(k)].gather(1, ck)
            if k == "M":
                bb = self.duals["b"]
                J = J + lam[:, 0] - phi_dot(bb, Xk[..., :N_NUM], yk)[:, 0] + (bb * self.rel["m_hat"]).sum(1) \
                    + self.rel["r_M"] * safe_norm(bb)
            else:
                J = J + (tv_worst_case(lam.detach(), self.rel["p_hat"], self.rel["r_TV"]) * lam).sum(1)
        return J
