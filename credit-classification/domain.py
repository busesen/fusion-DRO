"""The support set Xi = [-1, 1]^14 x (valid one-hot categories) x {0, 1} in network-input coordinates.

A point is a 59-vector x (14 numerical coordinates, then the one-hot blocks of the 9 categorical features) and a label.
The inner maximizations search over cases: a label alone (the moment set), or a histogram cell, which fixes the label,
sex, education, marital status, the band of PAY_0 and the band of the number of late months among PAY_2..PAY_6.
The ascent relaxes every one-hot block to its simplex over the levels the case allows; round_valid maps a relaxed
point to the nearest valid point of its case.
"""
from __future__ import annotations

import numpy as np
import torch

from credit_data import LEVELS, N_CELLS, N_IN, N_NUM

OFF = [int(o) for o in np.cumsum([0] + LEVELS)[:-1] + N_NUM]   # first column of each one-hot block
K_SEX, K_EDU, K_MAR, K_PAY0 = 0, 1, 2, 3
PAY_LATER = [4, 5, 6, 7, 8]
# the 45 one-hot columns as 9 blocks padded to 6 levels
_PAD_COL = np.array([[OFF[k] + min(a, LEVELS[k] - 1) for a in range(6)] for k in range(len(LEVELS))])
_PAD_OK = np.array([[a < LEVELS[k] for a in range(6)] for k in range(len(LEVELS))])
_UNPAD = np.array([6 * k + a for k in range(len(LEVELS)) for a in range(LEVELS[k])])
_CONST = {}


def _c(name, ref, value, dtype):
    """constants cached per device (a host-to-device copy inside a captured CUDA graph is not allowed)"""
    key = (name, ref.device, dtype)
    if key not in _CONST:
        _CONST[key] = torch.as_tensor(value, dtype=dtype, device=ref.device)
    return _CONST[key]


def cell_cases(dev):
    """(label, PAY_0 band, late-months band, (sex, education, marriage) codes, cell index) of all 432 cells"""
    c = torch.arange(N_CELLS, device=dev)
    c18, mar, edu, sex = c // 24, c % 3, (c // 3) % 4, (c // 12) % 2
    return c18 // 9, (c18 % 9) // 3, c18 % 3, torch.stack([sex, edu, mar], -1), c


def label_cases(dev):
    """the two labels with all categories free (-1)"""
    y = torch.arange(2, device=dev)
    m1 = torch.full_like(y, -1)
    return y, m1, m1, torch.full((2, 3), -1, device=dev), torch.zeros_like(y)


def _blocks(X):
    return X[..., _c("pad_col", X, _PAD_COL, torch.long)]


def _cumsum_last(u):
    """exact cumulative sum over a short last axis (faster than the GPU scan kernel for length 5 or 6)"""
    out = [u[..., :1]]
    for k in range(1, u.shape[-1]):
        out.append(out[-1] + u[..., k:k + 1])
    return torch.cat(out, -1)


SORT6 = [(0, 5), (1, 3), (2, 4), (1, 2), (3, 4), (0, 3), (2, 5), (0, 1), (2, 3), (4, 5), (1, 2), (3, 4)]


def _sort_desc6(v):
    """decreasing sort of a last axis of length 6 by a sorting network"""
    c = list(v.unbind(-1))
    for i, j in SORT6:
        c[i], c[j] = torch.maximum(c[i], c[j]), torch.minimum(c[i], c[j])
    return torch.stack(c, -1)


def _proj_simplex(v, mask):
    """Euclidean projection of the last axis onto the probability simplex over the allowed entries"""
    v = torch.where(mask, v, torch.full_like(v, -1e9))
    u = _sort_desc6(v)
    css = _cumsum_last(u) - 1.0
    ind = torch.arange(1, 7, device=v.device, dtype=v.dtype)
    rho = ((u - css / ind) > 0).sum(-1, keepdim=True).clamp_min(1)
    w = (v - css.gather(-1, rho - 1) / rho.to(v.dtype)).clamp_min(0.0)
    return torch.where(mask, w, torch.zeros_like(w))


def _block_mask(Rb, Sx, ref):
    """allowed levels of the padded blocks: SEX/EDUCATION/MARRIAGE at their codes Sx where >= 0, PAY_0 in band Rb"""
    lv = torch.arange(6, device=ref.device)
    fixed = (Sx[..., :, None] < 0) | (lv == Sx[..., :, None])
    band = torch.where(lv <= 2, 0, torch.where(lv == 3, 1, 2))
    pay0 = ((Rb[..., None] < 0) | (band == Rb[..., None]))[..., None, :]
    rest = torch.ones(Rb.shape + (5, 6), dtype=torch.bool, device=ref.device)
    return _c("pad_ok", ref, _PAD_OK, torch.bool) & torch.cat([fixed, pay0, rest], -2)


def project_relaxed(X, Rb, Sx):
    """numerical coordinates onto [-1, 1], every one-hot block onto its simplex over the allowed levels"""
    w = _proj_simplex(_blocks(X), _block_mask(Rb, Sx, X)).flatten(-2)[..., _c("unpad", X, _UNPAD, torch.long)]
    return torch.cat([X[..., :N_NUM].clamp(-1.0, 1.0), w], -1)


def round_valid(X, Rb, Fb, Sx):
    """Nearest valid point of the case: in every block the allowed level with the largest score; for PAY_2..PAY_6
    the best late and not-late level per month and the best admissible number of late months (exact)."""
    out = [X[..., :N_NUM].clamp(-1.0, 1.0)]
    blk = _blocks(X)
    codes = torch.where(_block_mask(Rb, Sx, X), blk, torch.full_like(blk, -1e9)).argmax(-1)
    for k in (K_SEX, K_EDU, K_MAR, K_PAY0):
        out.append(torch.nn.functional.one_hot(codes[..., k], LEVELS[k]).to(X.dtype))
    S = torch.stack([X[..., OFF[k]:OFF[k] + 6] for k in PAY_LATER], -2)
    s_non, i_non = S[..., :3].max(-1)
    s_late, i_late = S[..., 3:].max(-1)
    g_sorted, order = (s_late - s_non).sort(-1, descending=True)
    tot = torch.cat([torch.zeros_like(g_sorted[..., :1]), _cumsum_last(g_sorted)], -1)  # best total for k late months
    n_late = torch.arange(6, device=X.device)
    band = torch.where(n_late == 0, 0, torch.where(n_late <= 2, 1, 2))
    tot = torch.where((Fb[..., None] < 0) | (band == Fb[..., None]), tot, torch.full_like(tot, -1e18))
    rank = torch.empty_like(order).scatter_(-1, order, torch.arange(5, device=X.device).expand_as(order))
    code = torch.where(rank < tot.argmax(-1, keepdim=True), i_late + 3, i_non)
    out.append(torch.nn.functional.one_hot(code, 6).to(X.dtype).flatten(-2))
    return torch.cat(out, -1)


def random_valid(shape, Rb, Fb, Sx, gen, device, dtype):
    """random valid points of the cases (numerical coordinates uniform on [-1, 1])"""
    z = torch.rand(shape + (N_NUM,), generator=gen, device=device, dtype=dtype) * 2.0 - 1.0
    u = torch.rand(shape + (N_IN - N_NUM,), generator=gen, device=device, dtype=dtype)
    return round_valid(torch.cat([z, u], -1), Rb, Fb, Sx)


def perturb_valid(X, Rb, Fb, Sx, sigma, gen, tau=0.35):
    """nearby valid points: Gaussian noise on the numerical coordinates, Gumbel noise on the one-hot encodings"""
    z = X[..., :N_NUM] + sigma * torch.randn(X[..., :N_NUM].shape, generator=gen, device=X.device, dtype=X.dtype)
    u = torch.rand(X[..., N_NUM:].shape, generator=gen, device=X.device, dtype=X.dtype).clamp(1e-9, 1 - 1e-9)
    return round_valid(torch.cat([z, X[..., N_NUM:] - tau * torch.log(-torch.log(u))], -1), Rb, Fb, Sx)
