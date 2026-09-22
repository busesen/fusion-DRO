from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd

from coffee_common import str2bool
from final_test_plot_coffee import _highlight_oos_table, METHOD_ORDER

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NU_GRID = np.linspace(0.05, 0.95, 19).round(2).tolist()
REPO_ROOT_ENV = "IWB_REPO_ROOT"


def _normalize_repo_root(raw: str) -> Path:
    p = Path(raw).expanduser().resolve()
    if p.name == "coffee-newsvendor":
        return p.parent.resolve()
    return p


def _run(cmd, *, env_extra: dict[str, str] | None = None):
    print("Command:")
    print(" ".join(str(c) for c in cmd))
    env = os.environ.copy()
    if env_extra:
        env.update({str(k): str(v) for k, v in env_extra.items()})
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def _json_safe_args(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def _write_run_config(
    run_root: Path,
    *,
    args: argparse.Namespace,
    run_barycenter_fitted: str,
    run_rkhs: str,
    resume: bool,
    env_extra: dict[str, str],
) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "argv": [sys.executable, *sys.argv],
        "run_root": str(run_root.resolve()),
        "args": _json_safe_args(args),
        "effective": {
            "resume": bool(resume),
            "run_barycenter_fitted": str(run_barycenter_fitted),
            "run_rkhs": str(run_rkhs),
            "repo_root": str(env_extra.get(REPO_ROOT_ENV, "")),
            "gurobi_license_file": str(env_extra.get("IWB_GUROBI_LICENSE_FILE", "")),
            "mosek_license_file": str(env_extra.get("IWB_MOSEK_LICENSE_FILE", "")),
        },
    }
    (run_root / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _list_target_groups(processed_xlsx: Path) -> list[str]:
    group_meta = pd.read_excel(processed_xlsx, sheet_name="group_meta", engine="openpyxl")
    if "group_id" in group_meta.columns:
        vals = group_meta["group_id"].astype(str).tolist()
    elif "Group" in group_meta.columns:
        vals = group_meta["Group"].astype(str).tolist()
    else:
        weekly = pd.read_excel(processed_xlsx, sheet_name="weekly_groups", engine="openpyxl")
        if "group_id" not in weekly.columns:
            raise ValueError("Could not infer target groups from processed coffee workbook.")
        vals = weekly["group_id"].astype(str).unique().tolist()
    return sorted(dict.fromkeys(vals))


def _latest_radii_dir(radii_dir: Path) -> Path | None:
    radii_run_dirs = sorted([p for p in radii_dir.iterdir() if p.is_dir() and p.name.startswith("radii_run_")])
    return radii_run_dirs[-1] if radii_run_dirs else None


def _results_timestamp(path: str | Path) -> str | None:
    for candidate in [Path(path), *Path(path).parents]:
        match = re.search(r"(\d{8}_\d{6})(?:_\d+)?$", candidate.name)
        if match:
            return match.group(1)
    return None


def _selection_summary_path(outdir: str | Path) -> Path:
    outdir = Path(outdir)
    timestamp = _results_timestamp(outdir)
    if timestamp is None:
        return outdir / "selection_summary.xlsx"
    return outdir / f"selection_summary_{timestamp}.xlsx"


def _existing_selection_summary_path(outdir: str | Path) -> Path | None:
    outdir = Path(outdir)
    preferred = _selection_summary_path(outdir)
    if preferred.exists():
        return preferred
    legacy = outdir / "selection_summary.xlsx"
    if legacy.exists():
        return legacy
    matches = sorted(outdir.glob("selection_summary*.xlsx"))
    return matches[0] if matches else None


def _cleanup_selection_summaries(outdir: str | Path, keep_path: str | Path) -> None:
    outdir = Path(outdir)
    keep_path = Path(keep_path).resolve()
    for candidate in outdir.glob("selection_summary*.xlsx"):
        if candidate.resolve() == keep_path:
            continue
        if candidate.is_file():
            candidate.unlink()


def _resolve_rkhs_parallelism(
    *,
    target_workers: int,
    rkhs_parallel_workers: int,
    rkhs_max_active_sessions: int,
    run_rkhs: bool,
    multi_target: bool,
) -> tuple[int, int]:
    eff_target_workers = max(int(target_workers), 1)
    eff_rkhs_parallel_workers = max(int(rkhs_parallel_workers), 1)
    cap = max(int(rkhs_max_active_sessions), 0)
    if (not bool(run_rkhs)) or cap == 0:
        return eff_target_workers, eff_rkhs_parallel_workers
    if multi_target:
        eff_target_workers = min(eff_target_workers, cap)
        eff_rkhs_parallel_workers = min(eff_rkhs_parallel_workers, max(1, cap // eff_target_workers))
    else:
        eff_rkhs_parallel_workers = min(eff_rkhs_parallel_workers, cap)
    return eff_target_workers, eff_rkhs_parallel_workers


def _run_single_target(
    *,
    python_bin: str,
    processed_xlsx: Path,
    price_csv: Path,
    target_group: str,
    run_root: Path,
    k_start: int,
    k_max: int,
    target_weight: float,
    val_fraction: float,
    delta_grid: str,
    epsilon_grid: str,
    nu_grid: str,
    wass_p: float,
    w2_mc_samples: int,
    affine_static_center: str,
    barycenter_num_samples: int,
    rkhs_n_J: int,
    rkhs_is_herding: str,
    rkhs_T: int,
    rkhs_time_limit: int,
    rkhs_source_weights: str,
    rkhs_parallel_workers: int,
    rkhs_max_active_sessions: int,
    gurobi_license_file: str,
    mosek_license_file: str,
    seed: int,
    num_seeds: int,
    run_erm: str,
    run_aff: str,
    run_static: str,
    run_pooled: str,
    run_mixture: str,
    run_minkowski: str,
    run_barycenter_fitted: str,
    run_rkhs: str,
    cvxpy_solver: str,
    resume: bool,
    env_extra: dict[str, str] | None = None,
) -> None:
    schedule_dir = run_root / "schedule"
    radii_dir = run_root / "radii"
    solver_dir = run_root / "solver_validation"
    final_dir = run_root / "final_test"
    for d in [schedule_dir, radii_dir, solver_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)

    schedule_xlsx = schedule_dir / "coffee_schedule.xlsx"
    solver_csv = solver_dir / "eval_empirical_all.csv"
    final_summary = _selection_summary_path(final_dir)

    if resume and _existing_selection_summary_path(final_dir) is not None:
        print(f"[pipeline][resume] target={target_group}: final summary already exists, skipping target.", flush=True)
        return

    if resume and schedule_xlsx.exists():
        print(f"[pipeline][resume] target={target_group}: reusing existing schedule.", flush=True)
    else:
        _run([
            python_bin,
            str(SCRIPT_DIR / "build_coffee_schedule.py"),
            "--processed_path", str(processed_xlsx),
            "--outdir", str(schedule_dir),
            "--target_group", str(target_group),
            "--k_start", str(k_start),
            "--k_max", str(k_max),
            "--target_weight", str(target_weight),
        ], env_extra=env_extra)

    latest_radii = _latest_radii_dir(radii_dir) if resume else None
    if resume and latest_radii is not None and (latest_radii / "training_params.csv").exists():
        print(f"[pipeline][resume] target={target_group}: reusing existing radii at {latest_radii.name}.", flush=True)
    else:
        _run([
            python_bin,
            str(SCRIPT_DIR / "compute_k_ball_radii_coffee.py"),
            "--data_path", str(processed_xlsx),
            "--schedule_path", str(schedule_xlsx),
            "--outdir", str(radii_dir),
            "--val_fraction", str(val_fraction),
            "--delta_grid", str(delta_grid),
            "--epsilon_grid", str(epsilon_grid),
            "--seed", str(seed),
            "--wass_p", str(wass_p),
            "--w2_mc_samples", str(w2_mc_samples),
            "--affine_static_center", str(affine_static_center),
        ], env_extra=env_extra)
        latest_radii = _latest_radii_dir(radii_dir)

    if latest_radii is None:
        raise ValueError(f"No radii_run directory produced for target={target_group}")

    if resume and solver_csv.exists():
        print(f"[pipeline][resume] target={target_group}: resuming solver from existing eval CSV.", flush=True)

    _run([
        python_bin,
        str(SCRIPT_DIR / "run_coffee_solver_grid.py"),
        "--training_params_csv", str(latest_radii / "training_params.csv"),
        "--price_csv", str(price_csv),
        "--outdir", str(solver_dir),
        "--run_erm", str(run_erm),
        "--run_aff", str(run_aff),
        "--run_static", str(run_static),
        "--run_pooled", str(run_pooled),
        "--run_mixture", str(run_mixture),
        "--run_minkowski", str(run_minkowski),
        "--run_barycenter_fitted", str(run_barycenter_fitted),
        "--run_rkhs", str(run_rkhs),
        "--nu_grid", str(nu_grid),
        "--rkhs_source_weights", str(rkhs_source_weights),
        "--wass_p", str(wass_p),
        "--seed", str(seed),
        "--num_seeds", str(int(num_seeds)),
        "--barycenter_num_samples", str(barycenter_num_samples),
        "--barycenter_seed", str(seed),
        "--rkhs_n_J", str(int(rkhs_n_J)),
        "--rkhs_is_herding", str(rkhs_is_herding),
        "--rkhs_T", str(int(rkhs_T)),
        "--rkhs_time_limit", str(int(rkhs_time_limit)),
        "--rkhs_parallel_workers", str(int(rkhs_parallel_workers)),
        "--rkhs_max_active_sessions", str(int(rkhs_max_active_sessions)),
        "--gurobi_license_file", str(gurobi_license_file),
        "--mosek_license_file", str(mosek_license_file),
        "--cvxpy_solver", str(cvxpy_solver),
        "--resume", str(resume),
    ], env_extra=env_extra)

    _run([
        python_bin,
        str(SCRIPT_DIR / "final_test_plot_coffee.py"),
        "--eval_csv", str(solver_csv),
        "--outdir", str(final_dir),
        "--k_total_grid", "all",
    ], env_extra=env_extra)


def _aggregate_all_targets(run_root: Path, targets: list[str]) -> Path:
    final_root = run_root / "final_test"
    final_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in targets:
        target_final_dir = run_root / "targets" / str(target) / "final_test"
        summary_path = _existing_selection_summary_path(target_final_dir)
        if summary_path is None:
            raise ValueError(f"Missing per-target summary workbook under {target_final_dir}")
        df = pd.read_excel(summary_path, sheet_name="oos", engine="openpyxl")
        # The per-target OOS sheet uses merged cells for target/sources/K_total.
        # Pandas reads merged blocks as first value + blanks below, so forward-fill
        # before concatenating targets.
        for col in ["target", "sources", "K_total"]:
            if col in df.columns:
                df[col] = df[col].ffill()
        if not df.empty:
            rows.append(df)
    if not rows:
        raise ValueError("No per-target OOS summaries found to aggregate.")
    combined = pd.concat(rows, ignore_index=True)
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    combined["_method_rank"] = combined["method"].astype(str).map(method_rank).fillna(999)
    combined = combined.sort_values(["target", "sources", "K_total", "_method_rank"], kind="stable").drop(columns=["_method_rank"]).reset_index(drop=True)
    out_path = _selection_summary_path(final_root)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="oos", index=False)
    _highlight_oos_table(out_path, "oos")
    _cleanup_selection_summaries(final_root, out_path)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end coffee-shop pipeline: prepare data, build schedule, compute radii, solve grid, and summarize final test.")
    p.add_argument("--repo_root", type=str, default="", help="Optional fusion-DRO root override. By default, paths are resolved relative to this script.")
    p.add_argument("--raw_xlsx", type=str, default="", help="Optional raw coffee workbook override.")
    p.add_argument("--target_group", type=str, default="group_1", help="Held-out target group id.")
    p.add_argument("--k_start", type=int, choices=[5], default=5)
    p.add_argument("--k_max", type=int, choices=[5], default=5)
    p.add_argument("--target_weight", type=float, default=0.5)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--delta_grid", type=str, default="[0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]")
    p.add_argument("--epsilon_grid", type=str, default="[0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]")
    p.add_argument("--nu_grid", type=str, default=str(DEFAULT_NU_GRID))
    p.add_argument("--wass_p", type=float, choices=[2.0], default=2.0)
    p.add_argument("--w2_mc_samples", type=int, default=128)
    p.add_argument("--affine_static_center", type=str, default="gelbrich", choices=["gelbrich"])
    p.add_argument("--barycenter_num_samples", type=int, default=128)
    p.add_argument("--rkhs_n_J", type=int, default=2000)
    p.add_argument("--rkhs_is_herding", type=str, default="False")
    p.add_argument("--rkhs_T", type=int, default=50)
    p.add_argument("--rkhs_time_limit", type=int, default=3600)
    p.add_argument("--rkhs_source_weights", type=str, default="uniform", choices=["uniform"])
    p.add_argument("--gurobi_license_file", type=str, default="", help="Optional explicit Gurobi license file path for the RKHS benchmark.")
    p.add_argument("--mosek_license_file", type=str, default="", help="Optional explicit MOSEK license file path for MOSEK-backed CVXPY solves.")
    p.add_argument("--target_workers", type=int, default=1)
    p.add_argument("--rkhs_parallel_workers", type=int, default=1)
    p.add_argument("--rkhs_max_active_sessions", type=int, default=2, help="Cap on simultaneous RKHS/Gurobi solves. Default 2; set to 0 to disable throttling.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num_seeds", type=int, default=1)
    p.add_argument("--run_erm", type=str, default="True")
    p.add_argument("--run_aff", type=str, default="True")
    p.add_argument("--run_static", type=str, default="True")
    p.add_argument("--run_pooled", type=str, default="True")
    p.add_argument("--run_mixture", type=str, default="True")
    p.add_argument("--run_minkowski", type=str, default="True")
    p.add_argument("--run_barycenter_fitted", type=str, default="True")
    p.add_argument("--run_rkhs", type=str, default="False")
    p.add_argument("--cvxpy_solver", type=str, default="MOSEK")
    p.add_argument("--skip_prepare", type=str, default="False")
    p.add_argument("--outdir", type=str, default="")
    p.add_argument("--resume", type=str, default="False")
    args = p.parse_args()

    if str2bool(args.run_rkhs):
        from rkhs import check_rkhs_dependencies
        check_rkhs_dependencies()

    env_extra: dict[str, str] = {}
    if str(args.repo_root).strip():
        repo_root = _normalize_repo_root(str(args.repo_root))
        env_extra[REPO_ROOT_ENV] = str(repo_root)
        os.environ[REPO_ROOT_ENV] = str(repo_root)
    if str(args.gurobi_license_file).strip():
        env_extra["GRB_LICENSE_FILE"] = str(Path(str(args.gurobi_license_file)).expanduser())
        env_extra["IWB_GUROBI_LICENSE_FILE"] = str(Path(str(args.gurobi_license_file)).expanduser())
        os.environ["GRB_LICENSE_FILE"] = str(Path(str(args.gurobi_license_file)).expanduser())
        os.environ["IWB_GUROBI_LICENSE_FILE"] = str(Path(str(args.gurobi_license_file)).expanduser())
    if str(args.mosek_license_file).strip():
        env_extra["MOSEKLM_LICENSE_FILE"] = str(Path(str(args.mosek_license_file)).expanduser())
        env_extra["IWB_MOSEK_LICENSE_FILE"] = str(Path(str(args.mosek_license_file)).expanduser())
        os.environ["MOSEKLM_LICENSE_FILE"] = str(Path(str(args.mosek_license_file)).expanduser())
        os.environ["IWB_MOSEK_LICENSE_FILE"] = str(Path(str(args.mosek_license_file)).expanduser())

    run_barycenter_fitted = str(args.run_barycenter_fitted)
    run_rkhs = str(args.run_rkhs)


    resume = str2bool(args.resume)
    results_root = SCRIPT_DIR / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    if str(args.outdir).strip():
        run_root = Path(args.outdir)
        if not run_root.is_absolute():
            if resume:
                run_root = results_root / str(args.outdir).strip()
            else:
                run_root = results_root / f"{str(args.outdir).strip()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        if resume:
            raise ValueError("resume=True requires --outdir pointing to an existing run folder")
        run_root = results_root / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)
    _write_run_config(
        run_root,
        args=args,
        run_barycenter_fitted=run_barycenter_fitted,
        run_rkhs=run_rkhs,
        resume=resume,
        env_extra=env_extra,
    )

    data_dir = run_root / "data"
    final_dir = run_root / "final_test"
    for d in [data_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)

    processed_xlsx = data_dir / "coffee_weekly_groups.xlsx"
    price_csv = data_dir / "unit_price_by_coffee_size.csv"

    skip_prepare = str2bool(args.skip_prepare)
    if resume and processed_xlsx.exists() and price_csv.exists():
        print("[pipeline][resume] reusing existing prepared coffee data.", flush=True)
        skip_prepare = True

    if not skip_prepare:
        cmd = [sys.executable, str(SCRIPT_DIR / "prepare_coffee_data.py"), "--outdir", str(data_dir)]
        if str(args.raw_xlsx).strip():
            cmd += ["--raw_xlsx", str(args.raw_xlsx)]
        _run(cmd, env_extra=env_extra)
    else:
        if not processed_xlsx.exists() or not price_csv.exists():
            raise ValueError("skip_prepare=True but processed data files are missing")

    target_group_raw = str(args.target_group).strip()
    if target_group_raw.lower() == "all":
        targets = _list_target_groups(processed_xlsx)
        targets_root = run_root / "targets"
        targets_root.mkdir(parents=True, exist_ok=True)
        req_target_workers = max(int(args.target_workers), 1)
        req_rkhs_parallel_workers = max(int(args.rkhs_parallel_workers), 1)
        target_workers, rkhs_parallel_workers = _resolve_rkhs_parallelism(
            target_workers=req_target_workers,
            rkhs_parallel_workers=req_rkhs_parallel_workers,
            rkhs_max_active_sessions=int(args.rkhs_max_active_sessions),
            run_rkhs=str2bool(run_rkhs),
            multi_target=True,
        )
        if target_workers != req_target_workers or rkhs_parallel_workers != req_rkhs_parallel_workers:
            print(
                f"[pipeline] RKHS throttling applied: target_workers {req_target_workers} -> {target_workers}, "
                f"rkhs_parallel_workers {req_rkhs_parallel_workers} -> {rkhs_parallel_workers} "
                f"(rkhs_max_active_sessions={int(args.rkhs_max_active_sessions)}).",
                flush=True,
            )
        if target_workers == 1:
            for idx, target in enumerate(targets, start=1):
                print(f"[pipeline][target {idx}/{len(targets)}] {target}", flush=True)
                _run_single_target(
                    python_bin=sys.executable,
                    processed_xlsx=processed_xlsx,
                    price_csv=price_csv,
                    target_group=target,
                    run_root=targets_root / str(target),
                    k_start=int(args.k_start),
                    k_max=int(args.k_max),
                    target_weight=float(args.target_weight),
                    val_fraction=float(args.val_fraction),
                    delta_grid=str(args.delta_grid),
                    epsilon_grid=str(args.epsilon_grid),
                    nu_grid=str(args.nu_grid),
                    wass_p=float(args.wass_p),
                    w2_mc_samples=int(args.w2_mc_samples),
                    affine_static_center=str(args.affine_static_center),
                    barycenter_num_samples=int(args.barycenter_num_samples),
                    rkhs_n_J=int(args.rkhs_n_J),
                    rkhs_is_herding=str(args.rkhs_is_herding),
                    rkhs_T=int(args.rkhs_T),
                    rkhs_time_limit=int(args.rkhs_time_limit),
                    rkhs_source_weights=str(args.rkhs_source_weights),
                    rkhs_parallel_workers=int(rkhs_parallel_workers),
                    rkhs_max_active_sessions=int(args.rkhs_max_active_sessions),
                    gurobi_license_file=str(args.gurobi_license_file),
                    mosek_license_file=str(args.mosek_license_file),
                    seed=int(args.seed),
                    num_seeds=int(args.num_seeds),
                    run_erm=str(args.run_erm),
                    run_aff=str(args.run_aff),
                    run_static=str(args.run_static),
                    run_pooled=str(args.run_pooled),
                    run_mixture=str(args.run_mixture),
                    run_minkowski=str(args.run_minkowski),
                    run_barycenter_fitted=run_barycenter_fitted,
                    run_rkhs=run_rkhs,
                    cvxpy_solver=str(args.cvxpy_solver),
                    resume=resume,
                    env_extra=env_extra,
                )
        else:
            print(f"[pipeline] running {len(targets)} targets with {target_workers} worker(s)", flush=True)
            with ThreadPoolExecutor(max_workers=target_workers) as ex:
                future_map = {}
                for idx, target in enumerate(targets, start=1):
                    print(f"[pipeline][target {idx}/{len(targets)}] queued {target}", flush=True)
                    fut = ex.submit(
                        _run_single_target,
                        python_bin=sys.executable,
                        processed_xlsx=processed_xlsx,
                        price_csv=price_csv,
                        target_group=target,
                        run_root=targets_root / str(target),
                        k_start=int(args.k_start),
                        k_max=int(args.k_max),
                        target_weight=float(args.target_weight),
                        val_fraction=float(args.val_fraction),
                        delta_grid=str(args.delta_grid),
                        epsilon_grid=str(args.epsilon_grid),
                        nu_grid=str(args.nu_grid),
                        wass_p=float(args.wass_p),
                        w2_mc_samples=int(args.w2_mc_samples),
                        affine_static_center=str(args.affine_static_center),
                        barycenter_num_samples=int(args.barycenter_num_samples),
                        rkhs_n_J=int(args.rkhs_n_J),
                        rkhs_is_herding=str(args.rkhs_is_herding),
                        rkhs_T=int(args.rkhs_T),
                        rkhs_time_limit=int(args.rkhs_time_limit),
                        rkhs_source_weights=str(args.rkhs_source_weights),
                        rkhs_parallel_workers=int(rkhs_parallel_workers),
                        rkhs_max_active_sessions=int(args.rkhs_max_active_sessions),
                        gurobi_license_file=str(args.gurobi_license_file),
                        mosek_license_file=str(args.mosek_license_file),
                        seed=int(args.seed),
                        num_seeds=int(args.num_seeds),
                        run_erm=str(args.run_erm),
                        run_aff=str(args.run_aff),
                        run_static=str(args.run_static),
                        run_pooled=str(args.run_pooled),
                        run_mixture=str(args.run_mixture),
                        run_minkowski=str(args.run_minkowski),
                        run_barycenter_fitted=run_barycenter_fitted,
                        run_rkhs=run_rkhs,
                        cvxpy_solver=str(args.cvxpy_solver),
                        resume=resume,
                        env_extra=env_extra,
                    )
                    future_map[fut] = target
                for fut in as_completed(future_map):
                    target = future_map[fut]
                    fut.result()
                    print(f"[pipeline][target done] {target}", flush=True)
        combined_path = _aggregate_all_targets(run_root, targets)
        print(f"Combined OOS workbook: {combined_path}")
    else:
        _, rkhs_parallel_workers = _resolve_rkhs_parallelism(
            target_workers=1,
            rkhs_parallel_workers=int(args.rkhs_parallel_workers),
            rkhs_max_active_sessions=int(args.rkhs_max_active_sessions),
            run_rkhs=str2bool(run_rkhs),
            multi_target=False,
        )
        if rkhs_parallel_workers != max(int(args.rkhs_parallel_workers), 1):
            print(
                f"[pipeline] RKHS throttling applied: rkhs_parallel_workers {int(args.rkhs_parallel_workers)} -> {rkhs_parallel_workers} "
                f"(rkhs_max_active_sessions={int(args.rkhs_max_active_sessions)}).",
                flush=True,
            )
        _run_single_target(
            python_bin=sys.executable,
            processed_xlsx=processed_xlsx,
            price_csv=price_csv,
            target_group=target_group_raw,
            run_root=run_root,
            k_start=int(args.k_start),
            k_max=int(args.k_max),
            target_weight=float(args.target_weight),
            val_fraction=float(args.val_fraction),
            delta_grid=str(args.delta_grid),
            epsilon_grid=str(args.epsilon_grid),
            nu_grid=str(args.nu_grid),
            wass_p=float(args.wass_p),
            w2_mc_samples=int(args.w2_mc_samples),
            affine_static_center=str(args.affine_static_center),
            barycenter_num_samples=int(args.barycenter_num_samples),
            rkhs_n_J=int(args.rkhs_n_J),
            rkhs_is_herding=str(args.rkhs_is_herding),
            rkhs_T=int(args.rkhs_T),
            rkhs_time_limit=int(args.rkhs_time_limit),
            rkhs_source_weights=str(args.rkhs_source_weights),
            rkhs_parallel_workers=int(rkhs_parallel_workers),
            rkhs_max_active_sessions=int(args.rkhs_max_active_sessions),
            gurobi_license_file=str(args.gurobi_license_file),
            mosek_license_file=str(args.mosek_license_file),
            seed=int(args.seed),
            num_seeds=int(args.num_seeds),
            run_erm=str(args.run_erm),
            run_aff=str(args.run_aff),
            run_static=str(args.run_static),
            run_pooled=str(args.run_pooled),
            run_mixture=str(args.run_mixture),
            run_minkowski=str(args.run_minkowski),
            run_barycenter_fitted=run_barycenter_fitted,
            run_rkhs=run_rkhs,
            cvxpy_solver=str(args.cvxpy_solver),
            resume=resume,
            env_extra=env_extra,
        )

    print(f"Saved under {run_root}")


if __name__ == "__main__":
    main()
