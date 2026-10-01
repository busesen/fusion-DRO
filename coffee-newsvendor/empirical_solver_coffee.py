from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

import coffee_radii_utils as radii_utils
from coffee_common import (
    COUNT_COL,
    DEMAND_COLS,
    GROUP_COL,
    branch_matrices,
    expected_cost_from_counts,
    load_cost_params,
    newsvendor_cost_matrix,
    parse_float_list,
    parse_list,
    pooled_support,
    source_sample_list,
    str2bool,
    weighted_mixture_support,
)

SCRIPT_DIR = Path(__file__).resolve().parent
from rkhs import solve_rkhs

DEFAULT_MOSEK_LICENSE = Path.home() / "mosek" / "mosek.lic"
ALT_MOSEK_LICENSE_ENV = "IWB_MOSEK_LICENSE_FILE"


def _solve_prob(prob: "cp.Problem", solver: str, verbose: bool) -> None:
    """Solve with the requested solver; if it raises (e.g. numerical trouble at a
    zero radius), retry once with SCS instead of aborting the whole grid. If both
    fail the problem is left unsolved, so the row is recorded as failed (NaN
    costs) and is never selected by validation."""
    try:
        prob.solve(solver=solver, verbose=verbose)
        return
    except Exception as exc:  # noqa: BLE001
        first_exc = exc
    if str(solver).upper() != "SCS":
        try:
            prob.solve(solver="SCS", verbose=verbose)
            print(f"[empirical_solver_coffee] {solver} failed ({first_exc}); solved with SCS instead, status={prob.status}")
            return
        except Exception:  # noqa: BLE001
            pass
    print(f"[empirical_solver_coffee] solver failure ({solver}, SCS fallback): {first_exc}")


def _val(var: Any) -> Any:
    """JSON-friendly value of a cvxpy variable (None if unsolved). Used to store
    the dual multipliers gamma_k and decision-rule coefficients a_k, A_k so the
    worst-case distributions can be reconstructed later (e.g. for plotting)."""
    v = getattr(var, "value", None)
    if v is None:
        return None
    arr = np.asarray(v, dtype=float)
    return float(arr) if arr.ndim == 0 else arr.tolist()


def _status_ok(status: str | None) -> bool:
    return str(status) in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal", "optimal_inaccurate", "optimal_stagnation"}


def _ensure_mosek_license(mosek_license_file: str = "") -> None:
    if str(mosek_license_file).strip():
        os.environ["MOSEKLM_LICENSE_FILE"] = str(Path(str(mosek_license_file)).expanduser())
        return
    if os.environ.get("MOSEKLM_LICENSE_FILE"):
        return
    if os.environ.get(ALT_MOSEK_LICENSE_ENV):
        os.environ["MOSEKLM_LICENSE_FILE"] = str(Path(os.environ[ALT_MOSEK_LICENSE_ENV]).expanduser())
        return
    if DEFAULT_MOSEK_LICENSE.exists():
        os.environ["MOSEKLM_LICENSE_FILE"] = str(DEFAULT_MOSEK_LICENSE)


def _repeat_rows_by_count(df: pd.DataFrame, count_col: str) -> np.ndarray:
    counts = pd.to_numeric(df[count_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    x = df[DEMAND_COLS].to_numpy(dtype=float)
    rows: List[np.ndarray] = []
    for xi, c in zip(x, counts):
        if c <= 0.0:
            continue
        c_int = int(round(float(c)))
        if abs(float(c) - float(c_int)) > 1e-9:
            raise ValueError(f"Non-integer count encountered in {count_col}; got {c}")
        if c_int > 0:
            rows.extend([xi.copy()] * c_int)
    if len(rows) == 0:
        return np.zeros((0, len(DEMAND_COLS)), dtype=float)
    return np.vstack(rows)


def _load_train_samples(data_path: str, source_values: Sequence[str], train_count_col: str) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    df = pd.read_excel(data_path, sheet_name="weekly_groups", engine="openpyxl").copy()
    df[GROUP_COL] = df[GROUP_COL].astype(str)
    keep = {str(v) for v in source_values}
    df = df[df[GROUP_COL].isin(keep)].copy()
    for col in DEMAND_COLS + [train_count_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    samples = []
    for src in source_values:
        part = df[df[GROUP_COL].astype(str) == str(src)].copy()
        arr = _repeat_rows_by_count(part, train_count_col)
        if arr.shape[0] == 0:
            raise ValueError(f"No training rows for source {src} in {data_path}")
        samples.append(arr)
    return df, samples


def _evaluate_on_split(df: pd.DataFrame, w: np.ndarray, cost_params: Dict[str, np.ndarray], count_col: str) -> float:
    return expected_cost_from_counts(df=df, w=w, cost_params=cost_params, count_col=count_col)


def _make_method_block(status: str, objective: Optional[float], w: Optional[np.ndarray], runtime_sec: float, train_cost: float, val_cost: float, test_cost: float, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "success": bool(_status_ok(status)),
        "status": str(status),
        "objective_value": None if objective is None else float(objective),
        "runtime_sec": float(runtime_sec),
        "order_qty": [] if w is None else [float(v) for v in np.asarray(w, dtype=float).reshape(-1).tolist()],
        "train_expected_cost": float(train_cost),
        "validation_expected_cost": float(val_cost),
        "test_expected_cost": float(test_cost),
    }
    if w is not None:
        for i, val in enumerate(np.asarray(w, dtype=float).reshape(-1), start=1):
            out[f"w_{i}"] = float(val)
    if extra:
        out.update(extra)
    return out


def _solve_single_ball_w2(nominal_samples: np.ndarray, nominal_weights: np.ndarray, eps_cost: float, cost_params: Dict[str, np.ndarray], solver: str, verbose: bool) -> Tuple[str, Optional[float], Optional[np.ndarray], Dict[str, Any]]:
    support = np.asarray(nominal_samples, dtype=float)
    probs = np.asarray(nominal_weights, dtype=float).reshape(-1)
    probs = probs / probs.sum()
    M, d = support.shape
    A_X, A_Xi = branch_matrices(cost_params)

    w = cp.Variable(d)
    gamma = cp.Variable(nonneg=True)
    s = cp.Variable(M)

    constraints = [w >= 0]
    for ell in range(A_X.shape[0]):
        dvec = np.asarray(A_Xi[ell], dtype=float)
        cval = A_X[ell] @ w
        extra = 0.25 * cp.quad_over_lin(dvec, gamma)
        constraints.append(s >= cval + support @ dvec + extra)

    prob = cp.Problem(cp.Minimize(gamma * float(eps_cost) + probs @ s), constraints)
    _solve_prob(prob, solver=solver, verbose=verbose)
    aux = {"dual_gamma": _val(gamma)}
    return str(prob.status), None if prob.value is None else float(prob.value), None if w.value is None else np.asarray(w.value, dtype=float), aux


def _solve_minkowski_w2(samples_list: Sequence[np.ndarray], t_weights: Sequence[float], eps_cost: np.ndarray, cost_params: Dict[str, np.ndarray], solver: str, verbose: bool) -> Tuple[str, Optional[float], Optional[np.ndarray], Dict[str, Any]]:
    K = len(samples_list)
    Ns = [s.shape[0] for s in samples_list]
    d = samples_list[0].shape[1]
    A_X, A_Xi = branch_matrices(cost_params)

    w = cp.Variable(d)
    gamma = cp.Variable(K, nonneg=True)
    s = [cp.Variable(Nk) for Nk in Ns]
    constraints = [w >= 0]
    for k in range(K):
        tk = float(t_weights[k])
        for ell in range(A_X.shape[0]):
            dvec = tk * np.asarray(A_Xi[ell], dtype=float)
            cval = tk * (A_X[ell] @ w)
            extra = 0.25 * cp.quad_over_lin(dvec, gamma[k])
            constraints.append(s[k] >= cval + samples_list[k] @ dvec + extra)

    sample_term = cp.sum([cp.sum(s[k]) / float(Ns[k]) for k in range(K)])
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(gamma, eps_cost)) + sample_term), constraints)
    _solve_prob(prob, solver=solver, verbose=verbose)
    aux = {"dual_gamma": _val(gamma)}
    return str(prob.status), None if prob.value is None else float(prob.value), None if w.value is None else np.asarray(w.value, dtype=float), aux


def _solve_static_w2(samples_list: Sequence[np.ndarray], t_weights: Sequence[float], eps_cost: np.ndarray, cost_params: Dict[str, np.ndarray], solver: str, verbose: bool) -> Tuple[str, Optional[float], Optional[np.ndarray], Dict[str, Any]]:
    K = len(samples_list)
    Ns = [s.shape[0] for s in samples_list]
    d = samples_list[0].shape[1]
    A_X, A_Xi = branch_matrices(cost_params)

    w = cp.Variable(d)
    a = cp.Variable((K, d))
    gamma = cp.Variable(K, nonneg=True)
    s = [cp.Variable(Nk) for Nk in Ns]

    constraints = [w >= 0, cp.sum(a, axis=0) == 0]
    for k in range(K):
        tk = float(t_weights[k])
        for ell in range(A_X.shape[0]):
            dvec = tk * np.asarray(A_Xi[ell], dtype=float) + a[k, :]
            cval = tk * (A_X[ell] @ w)
            extra = 0.25 * cp.quad_over_lin(dvec, gamma[k])
            constraints.append(s[k] >= cval + samples_list[k] @ dvec + extra)

    sample_term = cp.sum([cp.sum(s[k]) / float(Ns[k]) for k in range(K)])
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(gamma, eps_cost)) + sample_term), constraints)
    _solve_prob(prob, solver=solver, verbose=verbose)
    aux = {"dual_gamma": _val(gamma), "dual_a": _val(a)}
    return str(prob.status), None if prob.value is None else float(prob.value), None if w.value is None else np.asarray(w.value, dtype=float), aux


def _solve_affine_w2(samples_list: Sequence[np.ndarray], t_weights: Sequence[float], eps_cost: np.ndarray, cost_params: Dict[str, np.ndarray], solver: str, verbose: bool, psd_shift: float = 1e-8) -> Tuple[str, Optional[float], Optional[np.ndarray], Dict[str, Any]]:
    K = len(samples_list)
    Ns = [s.shape[0] for s in samples_list]
    d = samples_list[0].shape[1]
    A_X, A_Xi = branch_matrices(cost_params)
    eye = np.eye(d)

    w = cp.Variable(d)
    a = cp.Variable((K, d))
    gamma = cp.Variable(K, nonneg=True)
    s = [cp.Variable(Nk) for Nk in Ns]
    A = [cp.Variable((d, d), symmetric=True) for _ in range(K)]

    constraints = [w >= 0, cp.sum(a, axis=0) == 0, sum(A) == 0]
    for k in range(K):
        tk = float(t_weights[k])
        constraints.append(gamma[k] >= psd_shift)
        for ell in range(A_X.shape[0]):
            dvec = tk * np.asarray(A_Xi[ell], dtype=float) + a[k, :]
            cval = tk * (A_X[ell] @ w)
            for i in range(Ns[k]):
                xi = np.asarray(samples_list[k][i], dtype=float)
                P = gamma[k] * eye - A[k]
                v = dvec + 2.0 * gamma[k] * xi
                scalar = s[k][i] - cval + gamma[k] * float(xi @ xi)
                block = cp.bmat([
                    [P, 0.5 * cp.reshape(v, (d, 1), order="F")],
                    [0.5 * cp.reshape(v, (1, d), order="F"), cp.reshape(scalar, (1, 1), order="F")],
                ])
                constraints.append(block >> 0)

    sample_term = cp.sum([cp.sum(s[k]) / float(Ns[k]) for k in range(K)])
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(gamma, eps_cost)) + sample_term), constraints)
    _solve_prob(prob, solver=solver, verbose=verbose)
    aux = {"dual_gamma": _val(gamma), "dual_a": _val(a), "dual_A": [_val(Ak) for Ak in A]}
    return str(prob.status), None if prob.value is None else float(prob.value), None if w.value is None else np.asarray(w.value, dtype=float), aux


def _resolve_rkhs_source_weights(t_weights, rkhs_source_weights):
    if rkhs_source_weights != "uniform":
        raise ValueError("The paper's RKHS benchmark uses uniform reference weights.")
    if not len(t_weights):
        raise ValueError("At least one reference distribution is required.")
    return "uniform", np.ones(len(t_weights), dtype=float) / len(t_weights)


def _barycenter_fitted_support(samples_list: Sequence[np.ndarray], t_weights: Sequence[float], wass_p: float, barycenter_num_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if abs(float(wass_p) - 2.0) > 1e-12:
        raise ValueError("barycenter_fitted is currently supported only for wass_p=2 in coffee_shop.")
    means = []
    covs = []
    for samples in samples_list:
        mu = np.mean(samples, axis=0)
        xc = samples - mu[None, :]
        cov = (xc.T @ xc) / float(samples.shape[0])
        means.append(mu)
        covs.append(cov)
    mu_bar, cov_bar, iters, rel = radii_utils.gelbrich_barycenter(means=means, covs=covs, weights=t_weights)
    rng = np.random.default_rng(int(seed))
    nominal_samples, nominal_weights = radii_utils.discretize_gaussian(mean=mu_bar, cov=cov_bar, num_samples=int(barycenter_num_samples), rng=rng)
    return nominal_samples, nominal_weights, {
        "barycenter_iters": int(iters),
        "barycenter_relerr": float(rel),
        "barycenter_num_samples": int(barycenter_num_samples),
    }


def solve_empirical_population_coffee(
    data_path: str,
    price_csv: str,
    source_values: Sequence[str],
    train_count_col: str,
    validation_count_col: str,
    test_count_col: str,
    epsilons: Sequence[float],
    t_weights: Sequence[float],
    epsilon_single: float,
    wass_p: float,
    barycenter_num_samples: int = 128,
    barycenter_seed: int = 1234,
    run_erm: bool = True,
    run_aff: bool = True,
    run_static: bool = True,
    run_pooled: bool = True,
    run_mixture: bool = True,
    run_minkowski: bool = True,
    run_barycenter_fitted: bool = True,
    run_rkhs: bool = False,
    rkhs_nu: float = 0.3,
    rkhs_source_weights: str = "uniform",
    rkhs_n_J: int = 2000,
    rkhs_is_herding: bool = False,
    rkhs_T: int = 50,
    rkhs_time_limit: int = 3600,
    gurobi_license_file: str = "",
    mosek_license_file: str = "",
    cvxpy_solver: str = "MOSEK",
    verbose: bool = False,
    benchmark_outdir: str = "",
) -> Dict[str, Any]:
    if float(wass_p) != 2.0:
        raise ValueError("The paper experiment supports only wass_p=2.")
    if len(source_values) == 0:
        raise ValueError("source_values must be non-empty")
    if len(t_weights) != len(source_values):
        raise ValueError("t_weights length mismatch")
    if len(epsilons) != len(source_values):
        raise ValueError("epsilons length mismatch")
    rkhs_weight_mode, rkhs_weight_values = _resolve_rkhs_source_weights(t_weights, rkhs_source_weights)

    df, samples_list = _load_train_samples(data_path=data_path, source_values=source_values, train_count_col=train_count_col)
    cost_params = load_cost_params(price_csv)
    t_weights = np.asarray(t_weights, dtype=float)
    t_weights = t_weights / t_weights.sum()
    eps_arr = np.asarray(epsilons, dtype=float)
    val_df = df.copy()
    test_df = df.copy()
    train_df = df.copy()
    validation_samples = _repeat_rows_by_count(val_df, validation_count_col)
    test_samples = _repeat_rows_by_count(test_df, test_count_col)

    pooled_nominal_samples, pooled_nominal_weights = pooled_support(samples_list)
    mix_nominal_samples, mix_nominal_weights = weighted_mixture_support(samples_list, t_weights)

    out: Dict[str, Any] = {
        "config": {
            "data_path": str(data_path),
            "price_csv": str(price_csv),
            "source_values": [str(v) for v in source_values],
            "train_count_col": str(train_count_col),
            "validation_count_col": str(validation_count_col),
            "test_count_col": str(test_count_col),
            "epsilons": [float(v) for v in eps_arr.tolist()],
            "t_weights": [float(v) for v in t_weights.tolist()],
            "epsilon_single": float(epsilon_single),
            "wass_p": float(wass_p),
            "barycenter_num_samples": int(barycenter_num_samples),
            "barycenter_seed": int(barycenter_seed),
            "cvxpy_solver": str(cvxpy_solver),
            "run_erm": bool(run_erm),
            "run_aff": bool(run_aff),
            "run_static": bool(run_static),
            "run_pooled": bool(run_pooled),
            "run_mixture": bool(run_mixture),
            "run_minkowski": bool(run_minkowski),
            "run_barycenter_fitted": bool(run_barycenter_fitted),
            "run_rkhs": bool(run_rkhs),
            "rkhs_nu": float(rkhs_nu),
            "rkhs_n_J": int(rkhs_n_J),
            "rkhs_is_herding": bool(rkhs_is_herding),
            "rkhs_T": int(rkhs_T),
            "rkhs_time_limit": int(rkhs_time_limit),
            "gurobi_license_file": str(gurobi_license_file),
            "mosek_license_file": str(mosek_license_file),
        }
    }

    if str(cvxpy_solver).strip().upper() == "MOSEK":
        _ensure_mosek_license(mosek_license_file=str(mosek_license_file))

    def _eval_block(status: str, obj: Optional[float], w_val: Optional[np.ndarray], t0: float, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        runtime = time.time() - t0
        if w_val is None:
            train_cost = float("nan")
            val_cost = float("nan")
            test_cost = float("nan")
        else:
            train_cost = _evaluate_on_split(train_df, w_val, cost_params, train_count_col)
            val_cost = _evaluate_on_split(val_df, w_val, cost_params, validation_count_col)
            test_cost = _evaluate_on_split(test_df, w_val, cost_params, test_count_col)
        return _make_method_block(status=status, objective=obj, w=w_val, runtime_sec=runtime, train_cost=train_cost, val_cost=val_cost, test_cost=test_cost, extra=extra)

    if run_erm:
        t0 = time.time()
        status, obj, w_val, aux = _solve_single_ball_w2(pooled_nominal_samples, pooled_nominal_weights, eps_cost=0.0, cost_params=cost_params, solver=cvxpy_solver, verbose=verbose)
        out["erm"] = _eval_block(status, obj, w_val, t0, extra={"epsilon": 0.0, "nominal": "pooled_empirical", **aux})

    if run_pooled:
        t0 = time.time()
        status, obj, w_val, aux = _solve_single_ball_w2(
            pooled_nominal_samples,
            pooled_nominal_weights,
            eps_cost=float(epsilon_single) ** 2,
            cost_params=cost_params,
            solver=cvxpy_solver,
            verbose=verbose,
        )
        extra = {"epsilon_pooled": float(epsilon_single), "epsilon_pooled_cost": float(epsilon_single) ** 2, **aux}
        out["pooled"] = _eval_block(status, obj, w_val, t0, extra=extra)

    if run_aff:
        t0 = time.time()
        eps_cost = eps_arr ** 2
        status, obj, w_val, aux = _solve_affine_w2(samples_list, t_weights, eps_cost, cost_params, solver=cvxpy_solver, verbose=verbose)
        extra = {"eps_cost": [float(v) for v in (eps_arr ** 2).tolist()], **aux}
        out["affine"] = _eval_block(status, obj, w_val, t0, extra=extra)

    if run_static:
        t0 = time.time()
        status, obj, w_val, aux = _solve_static_w2(samples_list, t_weights, eps_arr ** 2, cost_params, solver=cvxpy_solver, verbose=verbose)
        extra = {"eps_cost": [float(v) for v in (eps_arr ** 2).tolist()], **aux}
        out["static"] = _eval_block(status, obj, w_val, t0, extra=extra)

    if run_mixture:
        t0 = time.time()
        status, obj, w_val, aux = _solve_single_ball_w2(mix_nominal_samples, mix_nominal_weights, eps_cost=float(epsilon_single) ** 2, cost_params=cost_params, solver=cvxpy_solver, verbose=verbose)
        extra = {"epsilon_mix": float(epsilon_single), "epsilon_mix_cost": float(epsilon_single) ** 2, **aux}
        out["mixture"] = _eval_block(status, obj, w_val, t0, extra=extra)

    if run_minkowski:
        t0 = time.time()
        eps_mink = np.full(len(samples_list), float(epsilon_single), dtype=float)
        radius_mode = "common_scalar"
        status, obj, w_val, aux = _solve_minkowski_w2(samples_list, t_weights, eps_mink ** 2, cost_params, solver=cvxpy_solver, verbose=verbose)
        extra = {"eps_cost": [float(v) for v in (eps_mink ** 2).tolist()], "radius_mode": radius_mode, **aux}
        out["minkowski"] = _eval_block(status, obj, w_val, t0, extra=extra)

    if run_barycenter_fitted:
        t0 = time.time()
        bary_t0 = time.time()
        nominal_samples, nominal_weights, bary_meta = _barycenter_fitted_support(samples_list, t_weights, float(wass_p), int(barycenter_num_samples), int(barycenter_seed))
        bary_support_time_sec = float(time.time() - bary_t0)
        status, obj, w_val, aux = _solve_single_ball_w2(nominal_samples, nominal_weights, eps_cost=float(epsilon_single) ** 2, cost_params=cost_params, solver=cvxpy_solver, verbose=verbose)
        extra = {
            "epsilon": float(epsilon_single),
            "epsilon_cost": float(epsilon_single) ** 2,
            "additional_solve_time_sec": float(bary_support_time_sec),
            **bary_meta,
            **aux,
            # the sampled Gaussian-barycenter support, needed to reconstruct the worst case
            "nominal_support": np.asarray(nominal_samples, dtype=float).tolist(),
            "nominal_support_weights": np.asarray(nominal_weights, dtype=float).tolist(),
        }
        out["barycenter_fitted"] = _eval_block(status, obj, w_val, t0, extra=extra)

    if run_rkhs:
        t0 = time.time()
        status, obj, w_val, extra = solve_rkhs(
            samples_list,
            validation_samples,
            cost_params,
            benchmark_outdir or str(SCRIPT_DIR / "results"),
            verbose=verbose,
            seed=int(barycenter_seed),
            rkhs_nu=float(rkhs_nu),
            rkhs_source_weights=rkhs_weight_values,
            rkhs_source_weight_mode=rkhs_weight_mode,
            rkhs_n_J=int(rkhs_n_J),
            rkhs_is_herding=bool(rkhs_is_herding),
            rkhs_T=int(rkhs_T),
            rkhs_time_limit=int(rkhs_time_limit),
            gurobi_license_file=str(gurobi_license_file),
        )
        out["rkhs"] = _eval_block(status, obj, w_val, t0, extra=extra)

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Solve coffee-shop newsvendor DRO benchmarks on empirical source samples.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--price_csv", type=str, required=True)
    p.add_argument("--source_values", type=str, required=True)
    p.add_argument("--train_count_col", type=str, default="Count_train")
    p.add_argument("--validation_count_col", type=str, default="Count_target_val")
    p.add_argument("--test_count_col", type=str, default="Count_target_test")
    p.add_argument("--epsilons", type=str, default="[]")
    p.add_argument("--t_weights", type=str, required=True)
    p.add_argument("--epsilon_single", type=float, default=0.0)
    p.add_argument("--wass_p", type=float, choices=[2.0], default=2.0)
    p.add_argument("--barycenter_num_samples", type=int, default=128)
    p.add_argument("--barycenter_seed", type=int, default=1234)
    p.add_argument("--run_erm", type=str, default="True")
    p.add_argument("--run_aff", type=str, default="True")
    p.add_argument("--run_static", type=str, default="True")
    p.add_argument("--run_pooled", type=str, default="True")
    p.add_argument("--run_mixture", type=str, default="True")
    p.add_argument("--run_minkowski", type=str, default="True")
    p.add_argument("--run_barycenter_fitted", type=str, default="True")
    p.add_argument("--run_rkhs", type=str, default="False")
    p.add_argument("--rkhs_nu", type=float, default=0.3)
    p.add_argument("--rkhs_source_weights", type=str, default="uniform", choices=["uniform"])
    p.add_argument("--rkhs_n_J", type=int, default=2000)
    p.add_argument("--rkhs_is_herding", type=str, default="False")
    p.add_argument("--rkhs_T", type=int, default=50)
    p.add_argument("--rkhs_time_limit", type=int, default=3600)
    p.add_argument("--gurobi_license_file", type=str, default="")
    p.add_argument("--mosek_license_file", type=str, default="")
    p.add_argument("--cvxpy_solver", type=str, default="MOSEK")
    p.add_argument("--benchmark_outdir", type=str, default="")
    p.add_argument("--verbose", type=str, default="False")
    args = p.parse_args()

    run_rkhs = str2bool(args.run_rkhs)

    out = solve_empirical_population_coffee(
        data_path=args.data_path,
        price_csv=args.price_csv,
        source_values=parse_list(args.source_values),
        train_count_col=args.train_count_col,
        validation_count_col=args.validation_count_col,
        test_count_col=args.test_count_col,
        epsilons=parse_float_list(args.epsilons),
        t_weights=parse_float_list(args.t_weights),
        epsilon_single=float(args.epsilon_single),
        wass_p=float(args.wass_p),
        barycenter_num_samples=int(args.barycenter_num_samples),
        barycenter_seed=int(args.barycenter_seed),
        run_erm=str2bool(args.run_erm),
        run_aff=str2bool(args.run_aff),
        run_static=str2bool(args.run_static),
        run_pooled=str2bool(args.run_pooled),
        run_mixture=str2bool(args.run_mixture),
        run_minkowski=str2bool(args.run_minkowski),
        run_barycenter_fitted=str2bool(args.run_barycenter_fitted),
        run_rkhs=run_rkhs,
        rkhs_nu=float(args.rkhs_nu),
        rkhs_source_weights=str(args.rkhs_source_weights),
        rkhs_n_J=int(args.rkhs_n_J),
        rkhs_is_herding=str2bool(args.rkhs_is_herding),
        rkhs_T=int(args.rkhs_T),
        rkhs_time_limit=int(args.rkhs_time_limit),
        gurobi_license_file=str(args.gurobi_license_file),
        mosek_license_file=str(args.mosek_license_file),
        cvxpy_solver=str(args.cvxpy_solver),
        verbose=str2bool(args.verbose),
        benchmark_outdir=str(args.benchmark_outdir),
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
