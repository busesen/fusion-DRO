from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from coffee_common import parse_float_list, parse_list, str2bool
from empirical_solver_coffee import solve_empirical_population_coffee

COL_ROW_KIND = "row_kind"
COL_DELTA = "delta"
COL_EPSILON_SINGLE = "epsilon_single"
COL_RHO_K = "rho_k: W(ref,P_k)"
COL_EPSILON_K = "epsilon_k: (1+delta)*rho_k"

VAL_COL = "test_expected_cost (validation 10%)"
TEST_COL = "test_expected_cost (unseen 90%)"

METHOD_ORDER = ["erm", "affine", "static", "pooled", "mixture", "minkowski", "barycenter_fitted", "rkhs"]
DEFAULT_NU_GRID = np.linspace(0.05, 0.95, 19).round(2).tolist()
CONFIG_GROUP_COLS = [
    "K_total",
    "target_source",
    "selected_historical",
    "source_values",
    "source_roles",
    "t_weights",
    "training_data_path",
    "train_count_col",
    "validation_count_col",
    "test_count_col",
    "reference_name",
    "reference_meta",
    "wass_p",
]

ZERO_EPS_TOL = 1e-12
RESUME_FLOAT_TOL = 1e-10


def _parse_float_selector(raw: str) -> Tuple[bool, bool, List[float]]:
    s = str(raw).strip()
    if s == "":
        return False, False, []
    if s.lower() == "all":
        return True, True, []
    return True, False, [float(v) for v in parse_list(s)]


def _parse_int_selector(raw: str) -> Tuple[bool, List[int]]:
    s = str(raw).strip()
    if s == "" or s.lower() == "all":
        return True, []
    return False, [int(v) for v in parse_list(s)]


def _match_float(series: pd.Series, values: List[float], tol: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    keep = pd.Series(False, index=series.index)
    for v in values:
        keep = keep | ((s - float(v)).abs() <= float(tol))
    return keep


def _filter_rows(df: pd.DataFrame, k_selector_all: bool, k_values: List[int], delta_sel_provided: bool, delta_sel_all: bool, delta_values: List[float], delta_tol: float, eps_sel_provided: bool, eps_sel_all: bool, eps_values: List[float], eps_tol: float) -> pd.DataFrame:
    out = df.copy()
    if not k_selector_all:
        out = out[out["K_total"].astype(int).isin([int(v) for v in k_values])].copy()
    if delta_sel_provided and not delta_sel_all:
        aff_mask = out[COL_ROW_KIND].astype(str).eq("aff")
        keep = pd.Series(True, index=out.index)
        keep.loc[aff_mask] = _match_float(out.loc[aff_mask, COL_DELTA], delta_values, delta_tol)
        out = out.loc[keep].copy()
    if eps_sel_provided and not eps_sel_all:
        sb_mask = out[COL_ROW_KIND].astype(str).eq("single_ball")
        keep = pd.Series(True, index=out.index)
        keep.loc[sb_mask] = _match_float(out.loc[sb_mask, COL_EPSILON_SINGLE], eps_values, eps_tol)
        out = out.loc[keep].copy()
    return out.sort_values(["K_total", COL_ROW_KIND, COL_DELTA, COL_EPSILON_SINGLE], kind="stable").reset_index(drop=True)


def _extract_method_row(row_meta: Dict[str, Any], method: str, block: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row_meta)
    out["method"] = str(method)
    out["optimizer"] = "exact"
    out["empirical_success"] = block.get("success")
    out["empirical_status"] = block.get("status")
    out["empirical_objective_value"] = block.get("objective_value")
    out["empirical_runtime_sec"] = block.get("runtime_sec")
    out["train_expected_cost"] = block.get("train_expected_cost")
    out[VAL_COL] = block.get("validation_expected_cost")
    out[TEST_COL] = block.get("test_expected_cost")
    out["order_qty"] = json.dumps(block.get("order_qty", []))
    qty = block.get("order_qty", []) if isinstance(block.get("order_qty", []), list) else []
    for i, val in enumerate(qty, start=1):
        out[f"w_{i}"] = float(val)
    for key, value in block.items():
        if key in {"success", "status", "objective_value", "runtime_sec", "train_expected_cost", "validation_expected_cost", "test_expected_cost", "order_qty"}:
            continue
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value)
        else:
            out[key] = value
    return out


def _build_row_meta(row: pd.Series, row_kind: str) -> Dict[str, Any]:
    return {
        "row_kind": row_kind,
        "K_total": int(row["K_total"]),
        "target_source": str(row["target_source"]),
        "selected_historical": str(row["selected_historical"]),
        "source_values": str(row["source_values"]),
        "source_roles": str(row["source_roles"]),
        "t_weights": str(row["t_weights"]),
        "delta": row.get(COL_DELTA),
        "epsilon_single": row.get(COL_EPSILON_SINGLE),
        COL_RHO_K: row.get(COL_RHO_K, ""),
        COL_EPSILON_K: row.get(COL_EPSILON_K, ""),
        "training_data_path": str(row["training_data_path"]),
        "train_count_col": str(row["train_count_col"]),
        "validation_count_col": str(row["validation_count_col"]),
        "test_count_col": str(row["test_count_col"]),
        "reference_name": str(row.get("reference_name", "")),
        "reference_meta": str(row.get("reference_meta", "")),
        "shared_precompute_sec": row.get("shared_precompute_sec", np.nan),
        "wass_p": float(row.get("wass_p")),
        "nu": row.get("nu", np.nan),
    }


def _format_sources(raw_selected_historical: str) -> str:
    vals = parse_list(raw_selected_historical)
    return ", ".join(vals) if vals else "(none)"


def _equalish(a: Any, b: Any, tol: float = RESUME_FLOAT_TOL) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        af = float(a)
        bf = float(b)
        if np.isnan(af) and np.isnan(bf):
            return True
        return abs(af - bf) <= tol
    except Exception:
        return str(a) == str(b)


def _has_completed(existing: pd.DataFrame, *, method: str, row_meta: Dict[str, Any], run_seed: int) -> bool:
    if existing.empty:
        return False
    mask = existing["method"].astype(str).eq(str(method))
    mask &= existing["target_source"].astype(str).eq(str(row_meta["target_source"]))
    mask &= existing["selected_historical"].astype(str).eq(str(row_meta["selected_historical"]))
    mask &= pd.to_numeric(existing["K_total"], errors="coerce").fillna(-1).astype(int).eq(int(row_meta["K_total"]))
    mask &= pd.to_numeric(existing["run_seed"], errors="coerce").fillna(-1).astype(int).eq(int(run_seed))

    if method in {"affine", "static"}:
        mask &= existing["row_kind"].astype(str).eq("aff")
        mask &= pd.to_numeric(existing["delta"], errors="coerce").sub(float(row_meta["delta"])).abs().le(RESUME_FLOAT_TOL)
    elif method in {"pooled", "mixture", "barycenter_fitted"}:
        mask &= existing["row_kind"].astype(str).eq("single_ball")
        mask &= pd.to_numeric(existing["epsilon_single"], errors="coerce").sub(float(row_meta["epsilon_single"])).abs().le(RESUME_FLOAT_TOL)
    elif method == "minkowski":
        row_kind = str(row_meta.get("row_kind", ""))
        mask &= existing["row_kind"].astype(str).eq(row_kind)
        if row_kind == "aff":
            mask &= pd.to_numeric(existing["delta"], errors="coerce").sub(float(row_meta["delta"])).abs().le(RESUME_FLOAT_TOL)
        elif row_kind == "single_ball":
            mask &= pd.to_numeric(existing["epsilon_single"], errors="coerce").sub(float(row_meta["epsilon_single"])).abs().le(RESUME_FLOAT_TOL)
        else:
            return False
    elif method == "rkhs":
        mask &= pd.to_numeric(existing["nu"], errors="coerce").sub(float(row_meta["nu"])).abs().le(RESUME_FLOAT_TOL)
    elif method == "erm":
        mask &= existing["row_kind"].astype(str).eq("benchmark")
    else:
        return False
    return bool(mask.any())


def _sort_eval_df(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return eval_df
    eval_df = eval_df.copy()
    eval_df["_method_rank"] = eval_df["method"].astype(str).map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(999)
    sort_cols = ["target_source", "K_total", "selected_historical", "_method_rank", "run_seed", "delta", "epsilon_single", "nu"]
    return eval_df.sort_values(sort_cols, kind="stable").drop(columns=["_method_rank"]).reset_index(drop=True)


def _write_eval_outputs(outdir: Path, rows_out: List[Dict[str, Any]]) -> pd.DataFrame:
    eval_df = _sort_eval_df(pd.DataFrame(rows_out))
    latest_csv = outdir / "eval_empirical_all.csv"
    latest_xlsx = outdir / "eval_empirical_all.xlsx"
    eval_df.to_csv(latest_csv, index=False)
    with pd.ExcelWriter(latest_xlsx, engine="openpyxl") as writer:
        eval_df.to_excel(writer, sheet_name="eval", index=False)
    return eval_df


def _format_success_with_status(row_out: Dict[str, Any]) -> str:
    success = row_out.get("empirical_success")
    status = row_out.get("empirical_status")
    if bool(success):
        return "success=True"
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return "success=False"
    return f"success=False status={status}"


def _solve_rkhs_task(task: Dict[str, Any]) -> Dict[str, Any]:
    row_meta = dict(task["row_meta"])
    try:
        rkhs = solve_empirical_population_coffee(
            data_path=str(task["data_path"]),
            price_csv=str(task["price_csv"]),
            source_values=list(task["source_values"]),
            train_count_col=str(task["train_count_col"]),
            validation_count_col=str(task["validation_count_col"]),
            test_count_col=str(task["test_count_col"]),
            epsilons=[0.0] * len(task["source_values"]),
            t_weights=list(task["t_weights"]),
            epsilon_single=0.0,
            wass_p=float(task["wass_p"]),
            barycenter_num_samples=int(task["barycenter_num_samples"]),
            barycenter_seed=int(task["run_seed"]),
            run_erm=False,
            run_aff=False,
            run_static=False,
            run_pooled=False,
            run_mixture=False,
            run_minkowski=False,
            run_barycenter_fitted=False,
            run_rkhs=True,
            rkhs_nu=float(task["nu"]),
            rkhs_source_weights=str(task["rkhs_source_weights"]),
            rkhs_n_J=int(task["rkhs_n_J"]),
            rkhs_is_herding=bool(task["rkhs_is_herding"]),
            rkhs_T=int(task["rkhs_T"]),
            rkhs_time_limit=int(task["rkhs_time_limit"]),
            gurobi_license_file=str(task.get("gurobi_license_file", "")),
            mosek_license_file=str(task.get("mosek_license_file", "")),
            cvxpy_solver=str(task["cvxpy_solver"]),
            verbose=bool(task["verbose"]),
            benchmark_outdir=str(task["benchmark_outdir"]),
        )
        if "rkhs" not in rkhs:
            raise RuntimeError("rkhs solve returned no rkhs block")
        return _extract_method_row(row_meta, "rkhs", rkhs["rkhs"])
    except Exception as exc:
        block = {
            "success": False,
            "status": f"error: {exc}",
            "objective_value": np.nan,
            "runtime_sec": np.nan,
            "train_expected_cost": np.nan,
            "validation_expected_cost": np.nan,
            "test_expected_cost": np.nan,
            "order_qty": [],
            "nu": float(task["nu"]),
        }
        return _extract_method_row(row_meta, "rkhs", block)


def main() -> None:
    p = argparse.ArgumentParser(description="Run exact coffee-shop solver grid over affine and shared-epsilon training params.")
    p.add_argument("--training_params_csv", type=str, required=True)
    p.add_argument("--price_csv", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--k_total_grid", type=str, default="all")
    p.add_argument("--delta_grid", type=str, default="all")
    p.add_argument("--delta_tol", type=float, default=1e-10)
    p.add_argument("--epsilon_grid", type=str, default="all")
    p.add_argument("--epsilon_tol", type=float, default=1e-10)
    p.add_argument("--run_erm", type=str, default="True")
    p.add_argument("--run_aff", type=str, default="True")
    p.add_argument("--run_static", type=str, default="True")
    p.add_argument("--run_pooled", type=str, default="True")
    p.add_argument("--run_mixture", type=str, default="True")
    p.add_argument("--run_minkowski", type=str, default="True")
    p.add_argument("--run_barycenter_fitted", type=str, default="True")
    p.add_argument("--run_rkhs", type=str, default="False")
    p.add_argument("--nu_grid", type=str, default=str(DEFAULT_NU_GRID))
    p.add_argument("--rkhs_source_weights", type=str, default="uniform", choices=["uniform"])
    p.add_argument("--wass_p", type=float, choices=[2.0], default=2.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num_seeds", type=int, default=1)
    p.add_argument("--barycenter_num_samples", type=int, default=128)
    p.add_argument("--barycenter_seed", type=int, default=1234)
    p.add_argument("--rkhs_n_J", type=int, default=2000)
    p.add_argument("--rkhs_is_herding", type=str, default="False")
    p.add_argument("--rkhs_T", type=int, default=50)
    p.add_argument("--rkhs_time_limit", type=int, default=3600)
    p.add_argument("--rkhs_parallel_workers", type=int, default=1)
    p.add_argument("--rkhs_max_active_sessions", type=int, default=2)
    p.add_argument("--gurobi_license_file", type=str, default="")
    p.add_argument("--mosek_license_file", type=str, default="")
    p.add_argument("--cvxpy_solver", type=str, default="MOSEK")
    p.add_argument("--verbose", type=str, default="False")
    p.add_argument("--resume", type=str, default="False")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    resume = str2bool(args.resume)

    df = pd.read_csv(args.training_params_csv)
    k_selector_all, k_values = _parse_int_selector(args.k_total_grid)
    delta_sel_provided, delta_sel_all, delta_values = _parse_float_selector(args.delta_grid)
    eps_sel_provided, eps_sel_all, eps_values = _parse_float_selector(args.epsilon_grid)
    df = _filter_rows(df, k_selector_all, k_values, delta_sel_provided, delta_sel_all, delta_values, float(args.delta_tol), eps_sel_provided, eps_sel_all, eps_values, float(args.epsilon_tol))
    if df.empty:
        raise ValueError("No training-param rows remain after filtering")

    run_erm = str2bool(args.run_erm)
    run_aff = str2bool(args.run_aff)
    run_static = str2bool(args.run_static)
    run_pooled = str2bool(args.run_pooled)
    run_mixture = str2bool(args.run_mixture)
    run_minkowski = str2bool(args.run_minkowski)
    run_barycenter_fitted = str2bool(args.run_barycenter_fitted)
    run_rkhs = str2bool(args.run_rkhs)
    if run_rkhs:
        from rkhs import check_rkhs_dependencies
        check_rkhs_dependencies()
    nu_grid = [float(v) for v in parse_float_list(args.nu_grid)]
    if run_rkhs and len(nu_grid) == 0:
        raise ValueError("nu_grid must be non-empty when run_rkhs=True")
    rkhs_parallel_workers = max(int(args.rkhs_parallel_workers), 1)
    rkhs_max_active_sessions = max(int(args.rkhs_max_active_sessions), 0)
    if run_rkhs and rkhs_max_active_sessions > 0 and rkhs_parallel_workers > rkhs_max_active_sessions:
        print(
            f"[solver-grid] RKHS throttling applied: rkhs_parallel_workers {rkhs_parallel_workers} -> {rkhs_max_active_sessions} "
            f"(rkhs_max_active_sessions={rkhs_max_active_sessions}).",
            flush=True,
        )
        rkhs_parallel_workers = rkhs_max_active_sessions


    if int(args.num_seeds) <= 0:
        raise ValueError(f"num_seeds must be >= 1, got {args.num_seeds}")

    latest_csv = outdir / "eval_empirical_all.csv"
    existing_df = pd.DataFrame()
    rows_out: List[Dict[str, Any]] = []
    if resume and latest_csv.exists():
        existing_df = pd.read_csv(latest_csv)
        rows_out = existing_df.to_dict(orient="records")
        print(f"[solver-grid][resume] loaded {len(existing_df)} existing rows from {latest_csv}", flush=True)

    seed_list = [int(args.seed) + i for i in range(int(args.num_seeds))]
    base_seed = int(seed_list[0])
    grouped = list(df.groupby(CONFIG_GROUP_COLS, sort=False, dropna=False))

    for cfg_idx, (_, block) in enumerate(grouped, start=1):
        block = block.copy().reset_index(drop=True)
        first = block.iloc[0]
        target = str(first["target_source"])
        k_total = int(first["K_total"])
        hist_text = _format_sources(str(first["selected_historical"]))
        print(
            f"[solver-grid][config {cfg_idx}/{len(grouped)}] solving target: {target}, sources: {hist_text}, K={k_total}",
            flush=True,
        )

        aff_rows = block[block[COL_ROW_KIND].astype(str).eq("aff")].sort_values([COL_DELTA], kind="stable")
        sb_rows = block[block[COL_ROW_KIND].astype(str).eq("single_ball")].sort_values([COL_EPSILON_SINGLE], kind="stable")
        source_values = parse_list(first["source_values"])
        t_weights = [float(v) for v in json.loads(str(first["t_weights"]))]

        print(f"  [deterministic run seed = {base_seed}]", flush=True)

        for _, row in aff_rows.iterrows():
            row_meta = _build_row_meta(row, "aff")
            row_meta["run_seed"] = int(base_seed)
            row_meta["seed_idx"] = 1
            epsilons = [float(v) for v in json.loads(str(row[COL_EPSILON_K]))]
            if run_aff and _has_completed(existing_df, method="affine", row_meta=row_meta, run_seed=base_seed):
                print(f"    [affine] delta={float(row[COL_DELTA]):.6g} already done; skipping", flush=True)
            if run_static and _has_completed(existing_df, method="static", row_meta=row_meta, run_seed=base_seed):
                print(f"    [static] delta={float(row[COL_DELTA]):.6g} already done; skipping", flush=True)
            if ((not run_aff) or _has_completed(existing_df, method="affine", row_meta=row_meta, run_seed=base_seed)) and ((not run_static) or _has_completed(existing_df, method="static", row_meta=row_meta, run_seed=base_seed)):
                continue
            sol = solve_empirical_population_coffee(
                data_path=str(row["training_data_path"]),
                price_csv=args.price_csv,
                source_values=source_values,
                train_count_col=str(row["train_count_col"]),
                validation_count_col=str(row["validation_count_col"]),
                test_count_col=str(row["test_count_col"]),
                epsilons=epsilons,
                t_weights=t_weights,
                epsilon_single=0.0,
                wass_p=float(row_meta["wass_p"]),
                barycenter_num_samples=int(args.barycenter_num_samples),
                barycenter_seed=int(base_seed),
                run_erm=False,
                run_aff=bool(run_aff),
                run_static=bool(run_static),
                run_pooled=False,
                run_mixture=False,
                run_minkowski=False,
                run_barycenter_fitted=False,
                run_rkhs=False,
                rkhs_source_weights=str(args.rkhs_source_weights),
                cvxpy_solver=str(args.cvxpy_solver),
                verbose=str2bool(args.verbose),
                benchmark_outdir=str(outdir),
                rkhs_n_J=int(args.rkhs_n_J),
                rkhs_is_herding=str2bool(args.rkhs_is_herding),
                rkhs_T=int(args.rkhs_T),
                rkhs_time_limit=int(args.rkhs_time_limit),
                gurobi_license_file=str(args.gurobi_license_file),
                mosek_license_file=str(args.mosek_license_file),
            )
            if run_aff and "affine" in sol:
                rows_out.append(_extract_method_row(row_meta, "affine", sol["affine"]))
                existing_df = _write_eval_outputs(outdir, rows_out)
                print(f"    [affine] delta={float(row[COL_DELTA]):.6g} success={sol['affine'].get('success')}", flush=True)
            if run_static and "static" in sol:
                rows_out.append(_extract_method_row(row_meta, "static", sol["static"]))
                existing_df = _write_eval_outputs(outdir, rows_out)
                print(f"    [static] delta={float(row[COL_DELTA]):.6g} success={sol['static'].get('success')}", flush=True)

        for _, row in sb_rows.iterrows():
            row_meta = _build_row_meta(row, "single_ball")
            row_meta["run_seed"] = int(base_seed)
            row_meta["seed_idx"] = 1
            epsilon_single = float(row[COL_EPSILON_SINGLE])
            run_pooled_this = bool(run_pooled) and not (bool(run_erm) and abs(epsilon_single) <= ZERO_EPS_TOL)
            if run_pooled_this and _has_completed(existing_df, method="pooled", row_meta=row_meta, run_seed=base_seed):
                print(f"    [pooled] eps={epsilon_single:.6g} already done; skipping", flush=True)
            if run_mixture and _has_completed(existing_df, method="mixture", row_meta=row_meta, run_seed=base_seed):
                print(f"    [mixture] eps={epsilon_single:.6g} already done; skipping", flush=True)
            if run_minkowski and _has_completed(existing_df, method="minkowski", row_meta=row_meta, run_seed=base_seed):
                print(f"    [minkowski] eps={epsilon_single:.6g} already done; skipping", flush=True)
            if ((not run_pooled_this) or _has_completed(existing_df, method="pooled", row_meta=row_meta, run_seed=base_seed)) and ((not run_mixture) or _has_completed(existing_df, method="mixture", row_meta=row_meta, run_seed=base_seed)) and ((not run_minkowski) or _has_completed(existing_df, method="minkowski", row_meta=row_meta, run_seed=base_seed)):
                if not (bool(run_pooled) and abs(epsilon_single) <= ZERO_EPS_TOL and bool(run_erm)):
                    continue
            sol = solve_empirical_population_coffee(
                data_path=str(row["training_data_path"]),
                price_csv=args.price_csv,
                source_values=source_values,
                train_count_col=str(row["train_count_col"]),
                validation_count_col=str(row["validation_count_col"]),
                test_count_col=str(row["test_count_col"]),
                epsilons=[0.0] * len(source_values),
                t_weights=t_weights,
                epsilon_single=epsilon_single,
                wass_p=float(row_meta["wass_p"]),
                barycenter_num_samples=int(args.barycenter_num_samples),
                barycenter_seed=int(base_seed),
                run_erm=False,
                run_aff=False,
                run_static=False,
                run_pooled=run_pooled_this,
                run_mixture=bool(run_mixture),
                run_minkowski=bool(run_minkowski),
                run_barycenter_fitted=False,
                run_rkhs=False,
                rkhs_source_weights=str(args.rkhs_source_weights),
                cvxpy_solver=str(args.cvxpy_solver),
                verbose=str2bool(args.verbose),
                benchmark_outdir=str(outdir),
                rkhs_n_J=int(args.rkhs_n_J),
                rkhs_is_herding=str2bool(args.rkhs_is_herding),
                rkhs_T=int(args.rkhs_T),
                rkhs_time_limit=int(args.rkhs_time_limit),
                gurobi_license_file=str(args.gurobi_license_file),
                mosek_license_file=str(args.mosek_license_file),
            )
            if bool(run_pooled) and abs(epsilon_single) <= ZERO_EPS_TOL and bool(run_erm):
                print("    [pooled] eps=0 skipped because erm is run separately", flush=True)
            elif run_pooled_this and "pooled" in sol:
                rows_out.append(_extract_method_row(row_meta, "pooled", sol["pooled"]))
                existing_df = _write_eval_outputs(outdir, rows_out)
                print(f"    [pooled] eps={epsilon_single:.6g} success={sol['pooled'].get('success')}", flush=True)
            if run_mixture and "mixture" in sol:
                rows_out.append(_extract_method_row(row_meta, "mixture", sol["mixture"]))
                existing_df = _write_eval_outputs(outdir, rows_out)
                print(f"    [mixture] eps={epsilon_single:.6g} success={sol['mixture'].get('success')}", flush=True)
            if run_minkowski and "minkowski" in sol:
                rows_out.append(_extract_method_row(row_meta, "minkowski", sol["minkowski"]))
                existing_df = _write_eval_outputs(outdir, rows_out)
                print(f"    [minkowski] eps={epsilon_single:.6g} success={sol['minkowski'].get('success')}", flush=True)

        bench_meta = _build_row_meta(first, "benchmark")
        bench_meta["delta"] = np.nan
        bench_meta["epsilon_single"] = np.nan
        bench_meta["nu"] = np.nan
        bench_meta["run_seed"] = int(base_seed)
        bench_meta["seed_idx"] = 1
        if run_erm and _has_completed(existing_df, method="erm", row_meta=bench_meta, run_seed=base_seed):
            print("    [erm] already done; skipping", flush=True)
        elif run_erm:
            bench = solve_empirical_population_coffee(
                data_path=str(first["training_data_path"]),
                price_csv=args.price_csv,
                source_values=source_values,
                train_count_col=str(first["train_count_col"]),
                validation_count_col=str(first["validation_count_col"]),
                test_count_col=str(first["test_count_col"]),
                epsilons=[0.0] * len(source_values),
                t_weights=t_weights,
                epsilon_single=0.0,
                wass_p=float(first["wass_p"]),
                barycenter_num_samples=int(args.barycenter_num_samples),
                barycenter_seed=int(base_seed),
                run_erm=bool(run_erm),
                run_aff=False,
                run_static=False,
                run_pooled=False,
                run_mixture=False,
                run_minkowski=False,
                run_barycenter_fitted=False,
                run_rkhs=False,
                rkhs_source_weights=str(args.rkhs_source_weights),
                cvxpy_solver=str(args.cvxpy_solver),
                verbose=str2bool(args.verbose),
                benchmark_outdir=str(outdir),
                rkhs_n_J=int(args.rkhs_n_J),
                rkhs_is_herding=str2bool(args.rkhs_is_herding),
                rkhs_T=int(args.rkhs_T),
                rkhs_time_limit=int(args.rkhs_time_limit),
                gurobi_license_file=str(args.gurobi_license_file),
                mosek_license_file=str(args.mosek_license_file),
            )
            if "erm" in bench:
                rows_out.append(_extract_method_row(bench_meta, "erm", bench["erm"]))
                existing_df = _write_eval_outputs(outdir, rows_out)
                print(f"    [erm] success={bench['erm'].get('success')}", flush=True)

        stochastic_any = bool(run_barycenter_fitted or run_rkhs)
        if stochastic_any:
            for seed_idx, run_seed in enumerate(seed_list, start=1):
                print(f"  [stochastic seed {seed_idx}/{len(seed_list)} = {run_seed}]", flush=True)
                if run_barycenter_fitted:
                    for _, row in sb_rows.iterrows():
                        row_meta = _build_row_meta(row, "single_ball")
                        row_meta["run_seed"] = int(run_seed)
                        row_meta["seed_idx"] = int(seed_idx)
                        epsilon_single = float(row[COL_EPSILON_SINGLE])
                        if _has_completed(existing_df, method="barycenter_fitted", row_meta=row_meta, run_seed=run_seed):
                            print(f"    [barycenter_fitted] eps={epsilon_single:.6g} already done; skipping", flush=True)
                            continue
                        sol = solve_empirical_population_coffee(
                            data_path=str(row["training_data_path"]),
                            price_csv=args.price_csv,
                            source_values=source_values,
                            train_count_col=str(row["train_count_col"]),
                            validation_count_col=str(row["validation_count_col"]),
                            test_count_col=str(row["test_count_col"]),
                            epsilons=[0.0] * len(source_values),
                            t_weights=t_weights,
                            epsilon_single=epsilon_single,
                            wass_p=float(row_meta["wass_p"]),
                            barycenter_num_samples=int(args.barycenter_num_samples),
                            barycenter_seed=int(run_seed),
                            run_erm=False,
                            run_aff=False,
                            run_static=False,
                            run_pooled=False,
                            run_mixture=False,
                            run_minkowski=False,
                            run_barycenter_fitted=True,
                            run_rkhs=False,
                            rkhs_source_weights=str(args.rkhs_source_weights),
                            cvxpy_solver=str(args.cvxpy_solver),
                            verbose=str2bool(args.verbose),
                            benchmark_outdir=str(outdir),
                            rkhs_n_J=int(args.rkhs_n_J),
                            rkhs_is_herding=str2bool(args.rkhs_is_herding),
                            rkhs_T=int(args.rkhs_T),
                            rkhs_time_limit=int(args.rkhs_time_limit),
                            gurobi_license_file=str(args.gurobi_license_file),
                            mosek_license_file=str(args.mosek_license_file),
                        )
                        if "barycenter_fitted" in sol:
                            rows_out.append(_extract_method_row(row_meta, "barycenter_fitted", sol["barycenter_fitted"]))
                            existing_df = _write_eval_outputs(outdir, rows_out)
                            print(f"    [barycenter_fitted] eps={epsilon_single:.6g} success={sol['barycenter_fitted'].get('success')}", flush=True)
            if run_rkhs:
                rkhs_tasks: List[Dict[str, Any]] = []
                for seed_idx, run_seed in enumerate(seed_list, start=1):
                    for nu in nu_grid:
                        rkhs_meta = _build_row_meta(first, "benchmark")
                        rkhs_meta["delta"] = np.nan
                        rkhs_meta["epsilon_single"] = np.nan
                        rkhs_meta["nu"] = float(nu)
                        rkhs_meta["run_seed"] = int(run_seed)
                        rkhs_meta["seed_idx"] = int(seed_idx)
                        if _has_completed(existing_df, method="rkhs", row_meta=rkhs_meta, run_seed=run_seed):
                            print(f"    [rkhs] seed={run_seed} nu={float(nu):.6g} already done; skipping", flush=True)
                            continue
                        rkhs_tasks.append(
                            {
                                "row_meta": rkhs_meta,
                                "data_path": str(first["training_data_path"]),
                                "price_csv": str(args.price_csv),
                                "source_values": list(source_values),
                                "train_count_col": str(first["train_count_col"]),
                                "validation_count_col": str(first["validation_count_col"]),
                                "test_count_col": str(first["test_count_col"]),
                                "t_weights": list(t_weights),
                                "wass_p": float(first["wass_p"]),
                                "barycenter_num_samples": int(args.barycenter_num_samples),
                                "run_seed": int(run_seed),
                                "nu": float(nu),
                                "rkhs_source_weights": str(args.rkhs_source_weights),
                                "rkhs_n_J": int(args.rkhs_n_J),
                                "rkhs_is_herding": str2bool(args.rkhs_is_herding),
                                "rkhs_T": int(args.rkhs_T),
                                "rkhs_time_limit": int(args.rkhs_time_limit),
                                "gurobi_license_file": str(args.gurobi_license_file),
                                "mosek_license_file": str(args.mosek_license_file),
                                "cvxpy_solver": str(args.cvxpy_solver),
                                "verbose": str2bool(args.verbose),
                                "benchmark_outdir": str(outdir),
                            }
                        )
                if rkhs_tasks:
                    print(f"  [rkhs parallel] submitting {len(rkhs_tasks)} tasks with {rkhs_parallel_workers} worker(s)", flush=True)
                    if rkhs_parallel_workers == 1:
                        for task in rkhs_tasks:
                            row_out = _solve_rkhs_task(task)
                            rows_out.append(row_out)
                            existing_df = _write_eval_outputs(outdir, rows_out)
                            print(f"    [rkhs] seed={task['run_seed']} nu={float(task['nu']):.6g} {_format_success_with_status(row_out)}", flush=True)
                    else:
                        with ProcessPoolExecutor(max_workers=rkhs_parallel_workers) as ex:
                            future_map = {ex.submit(_solve_rkhs_task, task): task for task in rkhs_tasks}
                            for fut in as_completed(future_map):
                                task = future_map[fut]
                                row_out = fut.result()
                                rows_out.append(row_out)
                                existing_df = _write_eval_outputs(outdir, rows_out)
                                print(f"    [rkhs] seed={task['run_seed']} nu={float(task['nu']):.6g} {_format_success_with_status(row_out)}", flush=True)

    eval_df = pd.DataFrame(rows_out)
    if eval_df.empty:
        raise ValueError("No solver outputs were produced")
    eval_df = _sort_eval_df(eval_df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"eval_empirical_all_{ts}.csv"
    xlsx_path = outdir / f"eval_empirical_all_{ts}.xlsx"
    eval_df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        eval_df.to_excel(writer, sheet_name="eval", index=False)
    _write_eval_outputs(outdir, rows_out)

    print(f"CSV: {csv_path}")
    print(f"XLSX: {xlsx_path}")
    print(f"Saved under {outdir}")


if __name__ == "__main__":
    main()
