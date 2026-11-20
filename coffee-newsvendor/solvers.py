"""Conic reformulations of the 2-Wasserstein benchmarks of Section 5.2 and the Gelbrich
barycenter used to calibrate the radii and to center BARYCENTER."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from coffee_data import branch_matrices

# Extra keyword arguments for cvxpy's solve(), e.g. {"mosek_params": {"MSK_IPAR_NUM_THREADS": 1}}.
SOLVER_OPTS: Dict[str, Any] = {}
# Solver that produced the last solution ("MOSEK", or "SCS" if the requested solver failed).
LAST_SOLVER_USED: Dict[str, str] = {"name": ""}

Result = Tuple[str, Optional[float], Optional[np.ndarray], Dict[str, Any]]


def _solve_prob(prob: cp.Problem, solver: str, verbose: bool) -> None:
    """Solve with the requested solver; if it raises (e.g. numerical trouble at a zero
    radius), retry once with SCS. If both fail, the problem is left unsolved and the
    hyperparameter is never selected. A license error stops the run."""
    LAST_SOLVER_USED["name"] = ""
    try:
        prob.solve(solver=solver, verbose=verbose, **SOLVER_OPTS)
        LAST_SOLVER_USED["name"] = str(solver).upper()
        return
    except Exception as exc:  # noqa: BLE001
        first_exc = exc
    if "licen" in str(first_exc).lower():
        raise RuntimeError(f"{solver} license problem: {first_exc}") from first_exc
    if str(solver).upper() != "SCS":
        try:
            prob.solve(solver="SCS", verbose=verbose)
            LAST_SOLVER_USED["name"] = "SCS"
            return
        except Exception:  # noqa: BLE001
            pass
    print(f"[solvers] {solver} and SCS failed: {first_exc}")


def _val(var: Any) -> Any:
    v = getattr(var, "value", None)
    if v is None:
        return None
    arr = np.asarray(v, dtype=float)
    return float(arr) if arr.ndim == 0 else arr.tolist()


def _out(prob: cp.Problem, w: cp.Variable, aux: Dict[str, Any]) -> Result:
    return (str(prob.status), None if prob.value is None else float(prob.value),
            None if w.value is None else np.asarray(w.value, dtype=float), aux)


def solve_single_ball(support: np.ndarray, probs: np.ndarray, eps_cost: float, cost_params: Dict[str, np.ndarray],
                      solver: str, verbose: bool = False) -> Result:
    """One 2-Wasserstein ball with transport budget eps_cost = radius^2 around a discrete distribution."""
    support = np.asarray(support, dtype=float)
    probs = np.asarray(probs, dtype=float).reshape(-1)
    probs = probs / probs.sum()
    M, d = support.shape
    A_X, A_Xi = branch_matrices(cost_params)
    w, gamma, s = cp.Variable(d), cp.Variable(nonneg=True), cp.Variable(M)
    constraints = [w >= 0]
    for ell in range(A_X.shape[0]):
        dvec = A_Xi[ell]
        constraints.append(s >= A_X[ell] @ w + support @ dvec + 0.25 * cp.quad_over_lin(dvec, gamma))
    prob = cp.Problem(cp.Minimize(gamma * float(eps_cost) + probs @ s), constraints)
    _solve_prob(prob, solver, verbose)
    return _out(prob, w, {"dual_gamma": _val(gamma)})


def solve_minkowski(samples_list: Sequence[np.ndarray], t_weights: Sequence[float], eps_cost: np.ndarray,
                    cost_params: Dict[str, np.ndarray], solver: str, verbose: bool = False) -> Result:
    """Weighted Minkowski sum of K 2-Wasserstein balls with transport budgets eps_cost."""
    K, Ns, d = len(samples_list), [len(X) for X in samples_list], samples_list[0].shape[1]
    A_X, A_Xi = branch_matrices(cost_params)
    w, gamma = cp.Variable(d), cp.Variable(K, nonneg=True)
    s = [cp.Variable(N) for N in Ns]
    constraints = [w >= 0]
    for k in range(K):
        tk = float(t_weights[k])
        for ell in range(A_X.shape[0]):
            dvec = tk * A_Xi[ell]
            constraints.append(s[k] >= tk * (A_X[ell] @ w) + samples_list[k] @ dvec
                               + 0.25 * cp.quad_over_lin(dvec, gamma[k]))
    sample_term = cp.sum([cp.sum(s[k]) / float(Ns[k]) for k in range(K)])
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(gamma, eps_cost)) + sample_term), constraints)
    _solve_prob(prob, solver, verbose)
    return _out(prob, w, {"dual_gamma": _val(gamma)})


def solve_static(samples_list: Sequence[np.ndarray], t_weights: Sequence[float], eps_cost: np.ndarray,
                 cost_params: Dict[str, np.ndarray], solver: str, verbose: bool = False) -> Result:
    """Static decision rule approximation over the intersection of K 2-Wasserstein balls."""
    K, Ns, d = len(samples_list), [len(X) for X in samples_list], samples_list[0].shape[1]
    A_X, A_Xi = branch_matrices(cost_params)
    w, a, gamma = cp.Variable(d), cp.Variable((K, d)), cp.Variable(K, nonneg=True)
    s = [cp.Variable(N) for N in Ns]
    constraints = [w >= 0, cp.sum(a, axis=0) == 0]
    for k in range(K):
        tk = float(t_weights[k])
        for ell in range(A_X.shape[0]):
            dvec = tk * A_Xi[ell] + a[k, :]
            constraints.append(s[k] >= tk * (A_X[ell] @ w) + samples_list[k] @ dvec
                               + 0.25 * cp.quad_over_lin(dvec, gamma[k]))
    sample_term = cp.sum([cp.sum(s[k]) / float(Ns[k]) for k in range(K)])
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(gamma, eps_cost)) + sample_term), constraints)
    _solve_prob(prob, solver, verbose)
    return _out(prob, w, {"dual_gamma": _val(gamma), "dual_a": _val(a)})


def solve_static_sq(samples_list: Sequence[np.ndarray], t_weights: Sequence[float], eps_cost: np.ndarray,
                    cost_params: Dict[str, np.ndarray], solver: str, verbose: bool = False) -> Result:
    """Intermediate decision rules of Remark "Intermediate Decision Rules", not part of the table of
    Section 5.2 (see METHODS in run_newsvendor.py for how to run it). The affine decision rules with
    diagonal matrices A_k, i.e. corrections a_k^T xi + sum_j alpha_kj xi_j^2 with sum_k a_k = 0 and
    sum_k alpha_k = 0, which are the static decision rules applied to s(xi) = (xi, xi_1^2, ..., xi_m^2).
    For a piece with slope v = t_k d_l + a_k, the supremum over xi of
    v^T xi + sum_j alpha_kj xi_j^2 - gamma_k ||xi - xi_hat||^2 separates across j and equals
    -gamma_k ||xi_hat||^2 + sum_j (v_j + 2 gamma_k xi_hat_j)^2 / (4 (gamma_k - alpha_kj)),
    a second-order cone constraint. With alpha = 0 this is exactly solve_static."""
    K, Ns, d = len(samples_list), [len(X) for X in samples_list], samples_list[0].shape[1]
    A_X, A_Xi = branch_matrices(cost_params)
    w, a, gamma = cp.Variable(d), cp.Variable((K, d)), cp.Variable(K, nonneg=True)
    alpha = cp.Variable((K, d))
    s = [cp.Variable(N) for N in Ns]
    constraints = [w >= 0, cp.sum(a, axis=0) == 0, cp.sum(alpha, axis=0) == 0]
    for k in range(K):
        tk = float(t_weights[k])
        X = np.asarray(samples_list[k], dtype=float)
        sq_norms = np.sum(X ** 2, axis=1)
        for ell in range(A_X.shape[0]):
            dvec = tk * A_Xi[ell] + a[k, :]
            z = cp.Variable((Ns[k], d))  # z_ij >= (v_j + 2 gamma_k xi_ij)^2 / (4 (gamma_k - alpha_kj))
            for j in range(d):
                num = dvec[j] + 2.0 * gamma[k] * X[:, j]
                den = gamma[k] - alpha[k, j]
                # num^2 <= 4 den z  <=>  ||(num, z - den)|| <= z + den
                constraints.append(cp.SOC(z[:, j] + den, cp.vstack([num, z[:, j] - den]), axis=0))
            constraints.append(s[k] >= tk * (A_X[ell] @ w) - gamma[k] * sq_norms + cp.sum(z, axis=1))
    sample_term = cp.sum([cp.sum(s[k]) / float(Ns[k]) for k in range(K)])
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(gamma, eps_cost)) + sample_term), constraints)
    _solve_prob(prob, solver, verbose)
    return _out(prob, w, {"dual_gamma": _val(gamma), "dual_a": _val(a), "dual_alpha": _val(alpha)})


def solve_affine(samples_list: Sequence[np.ndarray], t_weights: Sequence[float], eps_cost: np.ndarray,
                 cost_params: Dict[str, np.ndarray], solver: str, verbose: bool = False,
                 psd_shift: float = 1e-8) -> Result:
    """Affine decision rule approximation over the intersection of K 2-Wasserstein balls (SDP)."""
    K, Ns, d = len(samples_list), [len(X) for X in samples_list], samples_list[0].shape[1]
    A_X, A_Xi = branch_matrices(cost_params)
    eye = np.eye(d)
    w, a, gamma = cp.Variable(d), cp.Variable((K, d)), cp.Variable(K, nonneg=True)
    s = [cp.Variable(N) for N in Ns]
    A = [cp.Variable((d, d), symmetric=True) for _ in range(K)]
    constraints = [w >= 0, cp.sum(a, axis=0) == 0, sum(A) == 0]
    for k in range(K):
        tk = float(t_weights[k])
        constraints.append(gamma[k] >= psd_shift)
        for ell in range(A_X.shape[0]):
            dvec = tk * A_Xi[ell] + a[k, :]
            cval = tk * (A_X[ell] @ w)
            for i in range(Ns[k]):
                xi = np.asarray(samples_list[k][i], dtype=float)
                v = dvec + 2.0 * gamma[k] * xi
                scalar = s[k][i] - cval + gamma[k] * float(xi @ xi)
                block = cp.bmat([
                    [gamma[k] * eye - A[k], 0.5 * cp.reshape(v, (d, 1), order="F")],
                    [0.5 * cp.reshape(v, (1, d), order="F"), cp.reshape(scalar, (1, 1), order="F")],
                ])
                constraints.append(block >> 0)
    sample_term = cp.sum([cp.sum(s[k]) / float(Ns[k]) for k in range(K)])
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(gamma, eps_cost)) + sample_term), constraints)
    _solve_prob(prob, solver, verbose)
    return _out(prob, w, {"dual_gamma": _val(gamma), "dual_a": _val(a), "dual_A": [_val(Ak) for Ak in A]})


# --------------------------------------------------------------------------- Gelbrich barycenter
def _sym(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.T)


def _sqrtm_psd(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w, v = np.linalg.eigh(_sym(mat))
    return v @ np.diag(np.sqrt(np.clip(w, 0.0, None) + eps)) @ v.T


def _invsqrtm_psd(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w, v = np.linalg.eigh(_sym(mat))
    return v @ np.diag(1.0 / np.sqrt(np.clip(w, eps, None))) @ v.T


def mean_and_covariance(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    xc = X - mu
    return mu, xc.T @ xc / float(len(X))


def gelbrich_barycenter(means: Sequence[np.ndarray], covs: Sequence[np.ndarray], weights: Sequence[float],
                        max_iter: int = 500, tol: float = 1e-9, reg: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and covariance matrix of the 2-Wasserstein barycenter of Gaussians (fixed-point iteration)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    mu_bar = sum(wk * np.asarray(mu, dtype=float) for wk, mu in zip(w, means))
    d = mu_bar.shape[0]
    eye = np.eye(d)
    sigma = _sym(sum(wk * _sym(np.asarray(c, dtype=float)) for wk, c in zip(w, covs))) + reg * eye
    for _ in range(max_iter):
        sigma_reg = _sym(sigma) + reg * eye
        s_sqrt, s_invsqrt = _sqrtm_psd(sigma_reg), _invsqrtm_psd(sigma_reg)
        accum = sum(wk * _sqrtm_psd(_sym(s_sqrt @ (_sym(np.asarray(c, dtype=float)) + reg * eye) @ s_sqrt))
                    for wk, c in zip(w, covs))
        sigma_next = _sym(s_invsqrt @ accum @ accum @ s_invsqrt)
        rel = np.linalg.norm(sigma_next - sigma, ord="fro") / max(float(np.linalg.norm(sigma, ord="fro")), 1e-16)
        sigma = sigma_next
        if rel <= tol:
            break
    return mu_bar, sigma


def gaussian_barycenter_samples(samples_list: Sequence[np.ndarray], weights: Sequence[float], num_samples: int,
                                seed: int, reg: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical distribution of num_samples i.i.d. samples from the Gelbrich (Gaussian) barycenter
    of the references' empirical means and covariance matrices."""
    stats = [mean_and_covariance(X) for X in samples_list]
    mu, cov = gelbrich_barycenter([m for m, _ in stats], [c for _, c in stats], weights)
    cov = _sym(cov) + reg * np.eye(mu.shape[0])
    rng = np.random.default_rng(int(seed))
    points = rng.multivariate_normal(mean=mu, cov=cov, size=int(num_samples), method="cholesky")
    return np.asarray(points, dtype=float), np.full(int(num_samples), 1.0 / num_samples)


def w2_distance(x: np.ndarray, p_x: np.ndarray, y: np.ndarray, p_y: np.ndarray) -> float:
    """2-Wasserstein distance between two discrete distributions (transport LP)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    cost = np.maximum(np.sum(x * x, 1)[:, None] + np.sum(y * y, 1)[None, :] - 2.0 * (x @ y.T), 0.0).reshape(-1)
    cols = np.arange(n * m)
    a_eq = coo_matrix((np.ones(2 * n * m), (np.concatenate([np.repeat(np.arange(n), m), n + np.tile(np.arange(m), n)]),
                                            np.concatenate([cols, cols]))), shape=(n + m, n * m)).tocsr()
    b_eq = np.concatenate([np.asarray(p_x, float) / np.sum(p_x), np.asarray(p_y, float) / np.sum(p_y)])
    res = linprog(c=cost, A_eq=a_eq, b_eq=b_eq, bounds=(0.0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"transport LP failed: {res.message}")
    return float(np.sqrt(max(res.fun, 0.0)))
