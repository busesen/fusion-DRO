"""Recover the worst-case distributions of the scaling experiment for one instance.

Rebuilds one (run, sweep, K, N) instance of a `scaling` result folder exactly as
runners/run_scaling.py does (same data, Gelbrich center and per-reference radii),
re-solves exact / affine / static / minkowski / mixture with the same formulations as
methods/*_w2.py, and keeps the dual quantities that locate the worst case:

  mixture, minkowski, static : multiplier gamma_k, decision-rule shift a_k and the
      constraint multipliers of every (sample, branch); the worst case moves sample
      xi to xi + d_{k,l} / (2 gamma_k), d_{k,l} = t_k b_l beta + a_k
  affine : additionally A_k; the worst case moves xi to (gamma_k I - A_k)^{-1} v / 2,
      v = d_{k,l} + 2 gamma_k xi (inner maximizer of the S-lemma block)
  exact  : multipliers p_{alpha,l} of the tuple constraints and the split w_{alpha,l,k};
      the worst case is the single distribution sum p_{alpha,l} delta_{xi*_{alpha,l}},
      xi*_{alpha,l} = xi_{k,alpha_k} + w_{alpha,l,k} / (2 lambda_k) (the same point for
      every k), i.e. the K samples of a tuple are transported to one common point.

Output: <outdir>/worst_case_run<r>_<sweep>_K<K>_N<N>.npz and a JSON summary comparing
the recomputed objectives with scaling.csv.

Run from fusion-DRO/:  python -m portfolio.plotting.worst_case_duals --results <folder>
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pandas as pd

from ..data_gen import generate_populations
from ..methods.shared import branch_factors, source_rule_weights
from ..runners import run_scaling as rs


def _cfg_from_results(folder: Path, solver: str) -> SimpleNamespace:
    cfg = json.loads((folder / "config.json").read_text())
    cfg["solver"] = solver
    cfg["verbose"] = False
    return SimpleNamespace(**cfg)


def build_instance(cfg, run: int, sweep: str, K: int, N: int):
    seed = int(cfg.seed_base) + int(run)
    if sweep == "N":
        job = rs._job_cfg(cfg, K_total=K, N_per_dist=N)
        pop = generate_populations(job, seed=seed)
        train = rs._all_source_train_data(pop)
        center, cw, meta = rs._sampled_w2_gaussian_barycenter_center(train, job)
    else:  # K sweep: prefix of the K_max family, center from the full family
        kmax = max(cfg.K_grid)
        full = rs._job_cfg(cfg, K_total=kmax, N_per_dist=int(cfg.N_for_K_sweep))
        pop_full = generate_populations(full, seed=seed)
        center, cw, meta = rs._sampled_w2_gaussian_barycenter_center(rs._all_source_train_data(pop_full), full)
        job = rs._job_cfg(cfg, K_total=K, N_per_dist=N)
        train = rs._all_source_train_data(rs._prefix_pop_data(pop_full, K))
    radii = rs._build_per_source_radii_from_fixed_center(train, job, delta=cfg.fixed_delta, center_samples=center,
                                                         center_weights=cw, center_meta=meta)
    return job, train, radii


def _solve(prob, solver):
    prob.solve(solver=solver)
    return prob.status


def solve_separable(samples, t, eps_cost, cfg, rule: str):
    """rule in {'minkowski', 'static', 'affine'}: K coupled single-ball problems."""
    K, d = len(samples), samples[0].shape[1]
    b, c = branch_factors(cfg.rho, cfg.eta)
    x, tau = cp.Variable(d), cp.Variable()
    gamma = cp.Variable(K, nonneg=True)
    s = [cp.Variable(len(X)) for X in samples]
    a = cp.Variable((K, d)) if rule in ("static", "affine") else None
    A = [cp.Variable((d, d), symmetric=True) for _ in range(K)] if rule == "affine" else None
    cons = [x >= 0, cp.sum(x) == 1]
    if a is not None:
        cons.append(cp.sum(a, axis=0) == 0)
    if A is not None:
        cons.append(sum(A) == 0)
    tagged = []   # (k, l, constraint)
    for k in range(K):
        if rule == "affine":
            cons.append(gamma[k] >= 1e-8)
        for l in range(2):
            dvec = t[k] * b[l] * x + (a[k, :] if a is not None else 0)
            cval = t[k] * c[l] * tau
            if rule != "affine":
                con = s[k] >= cval + samples[k] @ dvec + 0.25 * cp.quad_over_lin(dvec, gamma[k])
                cons.append(con)
                tagged.append((k, l, con))
            else:
                for i, xi in enumerate(samples[k]):
                    v = dvec + 2.0 * gamma[k] * xi
                    blk = cp.bmat([[gamma[k] * np.eye(d) - A[k], 0.5 * cp.reshape(v, (d, 1), order="F")],
                                   [0.5 * cp.reshape(v, (1, d), order="F"), cp.reshape(s[k][i] - cval + gamma[k] * float(xi @ xi), (1, 1), order="F")]])
                    con = blk >> 0
                    cons.append(con)
                    tagged.append((k, l, i, con))
    obj = cp.Minimize(cp.sum(cp.multiply(gamma, eps_cost)) + cp.sum([cp.sum(s[k]) / len(samples[k]) for k in range(K)]))
    prob = cp.Problem(obj, cons)
    status = _solve(prob, cfg.solver)
    beta, g = np.asarray(x.value), np.asarray(gamma.value)
    aval = np.zeros((K, d)) if a is None else np.asarray(a.value)
    Aval = np.zeros((K, d, d)) if A is None else np.stack([np.asarray(M.value) for M in A])
    # worst-case points and masses per (k, i, l)
    moved, mass = [], []
    for k in range(K):
        Nk = len(samples[k])
        mk = np.zeros((Nk, 2))
        pk = np.zeros((Nk, 2, d))
        for l in range(2):
            dvec = t[k] * b[l] * beta + aval[k]
            Minv = np.linalg.inv(g[k] * np.eye(d) - Aval[k])
            for i, xi in enumerate(samples[k]):
                pk[i, l] = 0.5 * Minv @ (dvec + 2.0 * g[k] * xi)
        if rule != "affine":
            for (kk, l, con) in tagged:
                if kk == k:
                    mk[:, l] = np.asarray(con.dual_value).reshape(-1)
        else:
            for (kk, l, i, con) in tagged:
                if kk == k:
                    mk[i, l] = float(np.asarray(con.dual_value)[d, d])
        moved.append(pk)
        mass.append(mk)
    return dict(status=status, objective=float(prob.value), beta=beta, tau=float(tau.value), gamma=g, a=aval, A=Aval,
                moved=moved, mass=mass)


def solve_mixture(samples, t, eps_radius, cfg):
    """Single ball around the weighted mixture with radius (sum_k t_k eps_k^2)^(1/2) (as methods/mixture_w2.py)."""
    K, d = len(samples), samples[0].shape[1]
    b, c = branch_factors(cfg.rho, cfg.eta)
    sup = np.vstack(samples)
    p = np.concatenate([np.full(len(X), t[k] / len(X)) for k, X in enumerate(samples)])
    p = p / p.sum()
    r = float(np.sqrt(np.dot(t, np.asarray(eps_radius, dtype=float) ** 2)))
    x, tau, gamma, s = cp.Variable(d), cp.Variable(), cp.Variable(nonneg=True), cp.Variable(len(sup))
    cons = [x >= 0, cp.sum(x) == 1]
    tagged = []
    for l in range(2):
        con = s >= c[l] * tau + sup @ (b[l] * x) + 0.25 * cp.quad_over_lin(b[l] * x, gamma)
        cons.append(con)
        tagged.append(con)
    prob = cp.Problem(cp.Minimize(gamma * r ** 2 + p @ s), cons)
    status = _solve(prob, cfg.solver)
    beta, g = np.asarray(x.value), float(gamma.value)
    moved_all = np.stack([sup + (b[l] * beta)[None, :] / (2 * g) for l in range(2)], axis=1)
    mass_all = np.stack([np.asarray(con.dual_value).reshape(-1) for con in tagged], axis=1)
    moved, mass, o = [], [], 0
    for X in samples:
        moved.append(moved_all[o:o + len(X)])
        mass.append(mass_all[o:o + len(X)])
        o += len(X)
    return dict(status=status, objective=float(prob.value), beta=beta, tau=float(tau.value), gamma=np.array([g]),
                radius=r, moved=moved, mass=mass)


def solve_exact(samples, eps_cost, cfg):
    K, d = len(samples), samples[0].shape[1]
    Ns = [len(X) for X in samples]
    b, c = branch_factors(cfg.rho, cfg.eta)
    alphas = np.array(list(product(*[range(n) for n in Ns])), dtype=int)
    x, tau = cp.Variable(d), cp.Variable()
    lam = cp.Variable(K, nonneg=True)
    gam = [cp.Variable(n) for n in Ns]
    W = cp.Variable((len(alphas) * 2 * K, d))
    cons = [x >= 0, cp.sum(x) == 1]
    tagged = []
    for ai, al in enumerate(alphas):
        for l in range(2):
            tt = 2 * ai + l
            lhs, rhs, wsum = c[l] * tau, 0, 0
            for k in range(K):
                wk = W[tt * K + k, :]
                lhs = lhs + samples[k][al[k]] @ wk + 0.25 * cp.quad_over_lin(wk, lam[k])
                rhs = rhs + gam[k][al[k]]
                wsum = wsum + wk
            con = lhs <= rhs
            cons += [con, wsum == b[l] * x]
            tagged.append((ai, l, con))
    obj = cp.Minimize(cp.sum(cp.multiply(lam, eps_cost)) + cp.sum([cp.sum(gam[k]) / Ns[k] for k in range(K)]))
    prob = cp.Problem(obj, cons)
    t0 = time.time()
    status = _solve(prob, cfg.solver)
    print(f"[exact] {status} in {time.time() - t0:.1f}s", flush=True)
    Wv, lv = np.asarray(W.value), np.asarray(lam.value)
    pts = np.zeros((len(alphas), 2, d))
    spread = np.zeros((len(alphas), 2))
    mass = np.zeros((len(alphas), 2))
    for ai, l, con in tagged:
        tt = 2 * ai + l
        cand = np.stack([samples[k][alphas[ai, k]] + Wv[tt * K + k] / (2 * lv[k]) for k in range(K)])
        pts[ai, l] = cand.mean(0)
        spread[ai, l] = np.max(np.linalg.norm(cand - cand.mean(0), axis=1))
        mass[ai, l] = float(np.ravel(con.dual_value)[0])
    return dict(status=status, objective=float(prob.value), beta=np.asarray(x.value), tau=float(tau.value), lam=lv,
                alphas=alphas, points=pts, mass=mass, point_spread=spread)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="scaling result folder (contains config.json, scaling.csv)")
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--sweep", choices=["N", "K"], default="N")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--solver", default="MOSEK")
    p.add_argument("--methods", default="mixture,minkowski,static,affine,exact")
    p.add_argument("--outdir", default="")
    a = p.parse_args()

    folder = Path(a.results)
    out = Path(a.outdir) if a.outdir else folder / "worst_case"
    out.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_results(folder, a.solver)
    job, train, radii = build_instance(cfg, a.run, a.sweep, a.K, a.N)
    samples = [np.asarray(X, float) for X in np.asarray(train["source_samples"], float)]
    t = source_rule_weights(job, train)
    eps_cost, eps_rad = np.asarray(radii["eps_cost"], float), np.asarray(radii["eps"], float)

    stem = f"worst_case_run{a.run}_{a.sweep}_K{a.K}_N{a.N}"
    save = {"samples": np.stack(samples), "t": t, "eps_cost": eps_cost, "eps_radius": eps_rad,
            "center": np.asarray(radii["sampled_center"], float), "rho": cfg.rho, "eta": cfg.eta}
    summary = {}
    for m in [s.strip() for s in a.methods.split(",") if s.strip()]:
        t0 = time.time()
        if m == "mixture":
            r = solve_mixture(samples, t, eps_rad, job)
        elif m in ("minkowski", "static", "affine"):
            r = solve_separable(samples, t, eps_cost, job, m)
        elif m == "exact":
            r = solve_exact(samples, eps_cost, job)
        else:
            raise ValueError(m)
        summary[m] = {"status": r["status"], "objective": r["objective"], "time_sec": time.time() - t0}
        print(f"{m:10s} {r['status']:10s} objective {r['objective']:.6f}  ({time.time() - t0:.1f}s)", flush=True)
        for key, val in r.items():
            if key in ("status",):
                continue
            if isinstance(val, list):
                val = np.stack(val)
            save[f"{m}__{key}"] = np.asarray(val)
        np.savez_compressed(out / f"{stem}.npz", **save)     # save after every method

    ref = pd.read_csv(folder / "scaling.csv")
    ref = ref[(ref.run == a.run) & (ref.sweep_axis == a.sweep) & (ref.K_total == a.K) & (ref.N_per_dist == a.N)]
    for m in summary:
        row = ref[ref.method == m]
        summary[m]["objective_in_scaling_csv"] = None if row.empty else float(row.objective.iloc[0])
    (out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
