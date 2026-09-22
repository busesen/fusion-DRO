from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from coffee_common import COUNT_COL, DEMAND_COLS, GROUP_COL, parse_float_list, parse_list
import coffee_radii_utils as radii_utils

COL_ROW_KIND = "row_kind"
COL_DELTA = "delta"
COL_EPSILON_SINGLE = "epsilon_single"
COL_RHO_K = "rho_k: W(ref,P_k)"
COL_EPSILON_K = "epsilon_k: (1+delta)*rho_k"


def _pairwise_cost_power(x, y, wass_p):
    if float(wass_p) != 2.0:
        raise ValueError("The paper experiment supports only wass_p=2.")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x2 = np.sum(x * x, axis=1)[:, None]
    y2 = np.sum(y * y, axis=1)[None, :]
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


def _discrete_ot_distance(x: np.ndarray, p_x: np.ndarray, y: np.ndarray, p_y: np.ndarray, wass_p: float) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p_x = np.asarray(p_x, dtype=float)
    p_y = np.asarray(p_y, dtype=float)
    n, m = int(x.shape[0]), int(y.shape[0])
    c = _pairwise_cost_power(x, y, wass_p=float(wass_p)).reshape(-1)
    nvar = n * m

    rows = []
    cols = []
    data = []
    b_eq: List[float] = []

    ii = np.repeat(np.arange(n), m)
    jj = np.arange(nvar)
    rows.append(ii)
    cols.append(jj)
    data.append(np.ones(nvar, dtype=float))
    b_eq.extend((p_x / float(p_x.sum())).tolist())

    row_base = n
    ii2 = row_base + np.tile(np.arange(m), n)
    rows.append(ii2)
    cols.append(jj)
    data.append(np.ones(nvar, dtype=float))
    b_eq.extend((p_y / float(p_y.sum())).tolist())

    a_eq = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n + m, nvar),
    ).tocsr()

    res = linprog(c=c, A_eq=a_eq, b_eq=np.asarray(b_eq, dtype=float), bounds=(0.0, None), method="highs")
    if not bool(res.success):
        raise RuntimeError(f"OT LP failed: status={res.status} message={res.message}")
    cost = float(max(res.fun, 0.0))
    return float(np.sqrt(cost))


def _reference_distribution(
    stats_list: List[radii_utils.EmpiricalStats],
    weights: List[float],
    wass_p: float,
    rng: np.random.Generator,
    w2_mc_samples: int,
    affine_static_center: str,
) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]:
    center_mode = str(affine_static_center).strip().lower()


    if center_mode != "gelbrich":
        raise ValueError(f"Unknown affine_static_center: {affine_static_center}")

    means = [s.mean for s in stats_list]
    covs = [s.cov for s in stats_list]
    mu_bar, cov_bar, bary_iters, bary_rel = radii_utils.gelbrich_barycenter(means=means, covs=covs, weights=weights)
    ref_points, ref_probs = radii_utils.discretize_gaussian(mean=mu_bar, cov=cov_bar, num_samples=int(w2_mc_samples), rng=rng)
    return ref_points, ref_probs, "gelbrich_barycenter", {
        "reference_type": "gelbrich_barycenter_gaussian_mc",
        "center_mode": "gelbrich",
        "barycenter_iters": int(bary_iters),
        "barycenter_relerr": float(bary_rel),
    }


def run_pipeline(
    data_path: str,
    schedule_path: str,
    outdir: str,
    schedule_sheet: str,
    val_fraction: float,
    delta_grid: List[float],
    epsilon_grid: List[float],
    seed: int,
    wass_p: float,
    w2_mc_samples: int,
    affine_static_center: str,
) -> Dict[str, Path]:
    rng = np.random.default_rng(int(seed))

    df = pd.read_excel(data_path, sheet_name="weekly_groups", engine="openpyxl").copy()
    df[GROUP_COL] = df[GROUP_COL].astype(str)
    df[COUNT_COL] = pd.to_numeric(df[COUNT_COL], errors="coerce").fillna(0.0)
    for col in DEMAND_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    schedule_df = pd.read_excel(schedule_path, sheet_name=schedule_sheet, engine="openpyxl").copy()
    schedule_df["target_source"] = schedule_df["target_source"].astype(str)
    targets = sorted(schedule_df["target_source"].unique().tolist())
    if len(targets) != 1:
        raise ValueError(f"Schedule must contain exactly one target, got {targets}")
    target = str(targets[0])

    target_df = df[df[GROUP_COL] == target].copy()
    target_pos_df, target_val_df, target_test_df = radii_utils.split_target_validation_test(
        target_df=target_df,
        count_col=COUNT_COL,
        val_fraction=float(val_fraction),
        rng=rng,
    )

    train_df = df.copy()
    train_df["Count_train"] = pd.to_numeric(train_df[COUNT_COL], errors="coerce").fillna(0.0)
    train_df["Count_target_val"] = 0.0
    train_df["Count_target_test"] = 0.0
    mask_target = train_df[GROUP_COL] == target
    train_df.loc[mask_target, "Count_train"] = 0.0
    train_df.loc[mask_target, "Count_target_test"] = pd.to_numeric(train_df.loc[mask_target, COUNT_COL], errors="coerce").fillna(0.0)
    train_df.loc[target_val_df.index, "Count_train"] = pd.to_numeric(train_df.loc[target_val_df.index, COUNT_COL], errors="coerce").fillna(0.0)
    train_df.loc[target_val_df.index, "Count_target_val"] = pd.to_numeric(train_df.loc[target_val_df.index, COUNT_COL], errors="coerce").fillna(0.0)
    train_df.loc[target_val_df.index, "Count_target_test"] = 0.0
    train_df.loc[target_test_df.index, "Count_target_test"] = pd.to_numeric(train_df.loc[target_test_df.index, COUNT_COL], errors="coerce").fillna(0.0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(outdir) / f"radii_run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    training_data_xlsx = run_dir / "coffee_training_data.xlsx"
    training_data_csv = run_dir / "coffee_training_data.csv"
    train_df.to_excel(training_data_xlsx, sheet_name="weekly_groups", index=False)
    train_df.to_csv(training_data_csv, index=False)

    rows_radii: List[Dict[str, Any]] = []
    rows_train: List[Dict[str, Any]] = []

    for _, row in schedule_df.iterrows():
        k_total = int(row["K_total"])
        selected_hist = parse_list(row["selected_historical"])
        source_values = parse_list(row["source_values"])
        source_roles = parse_list(row["source_roles"])
        t_weights = [float(v) for v in json.loads(str(row["t_weights"]))]

        frames = []
        stats_list: List[radii_utils.EmpiricalStats] = []
        for src, role, wk in zip(source_values, source_roles, t_weights):
            if role == "target_validation":
                src_df = train_df.loc[target_val_df.index].copy()
            else:
                src_df = train_df[train_df[GROUP_COL].astype(str) == str(src)].copy()
            stats = radii_utils.weighted_empirical_stats(
                source_name=str(src),
                role=str(role),
                df=src_df,
                context_cols=DEMAND_COLS,
                count_col="Count_train",
            )
            stats_list.append(stats)
            frames.append(src_df)
            rows_radii.append(
                {
                    "K_total": int(k_total),
                    "source": str(src),
                    "role": str(role),
                    "weight_wk": float(wk),
                    "support_rows": int(stats.support_rows),
                    "total_mass": float(stats.total_mass),
                    "target_source": target,
                }
            )

        precompute_t0 = datetime.now().timestamp()
        ref_points, ref_probs, ref_name, ref_meta = _reference_distribution(
            stats_list=stats_list,
            weights=t_weights,
            wass_p=float(wass_p),
            rng=rng,
            w2_mc_samples=int(w2_mc_samples),
            affine_static_center=str(affine_static_center),
        )

        rho_k = [
            float(_discrete_ot_distance(ref_points, ref_probs, st.points, st.probs, wass_p=float(wass_p)))
            for st in stats_list
        ]
        shared_precompute_sec = float(datetime.now().timestamp() - precompute_t0)

        for delta in delta_grid:
            rows_train.append(
                {
                    COL_ROW_KIND: "aff",
                    "K_total": int(k_total),
                    COL_DELTA: float(delta),
                    COL_EPSILON_SINGLE: np.nan,
                    "target_source": target,
                    "selected_historical": json.dumps(selected_hist),
                    "source_values": json.dumps(source_values),
                    "source_roles": json.dumps(source_roles),
                    "t_weights": json.dumps([float(v) for v in t_weights]),
                    COL_RHO_K: json.dumps([float(v) for v in rho_k]),
                    COL_EPSILON_K: json.dumps([float((1.0 + float(delta)) * v) for v in rho_k]),
                    "training_data_path": str(training_data_xlsx),
                    "train_count_col": "Count_train",
                    "validation_count_col": "Count_target_val",
                    "test_count_col": "Count_target_test",
                    "reference_name": ref_name,
                    "reference_meta": json.dumps(ref_meta),
                    "shared_precompute_sec": float(shared_precompute_sec),
                    "wass_p": float(wass_p),
                }
            )

        for eps in epsilon_grid:
            rows_train.append(
                {
                    COL_ROW_KIND: "single_ball",
                    "K_total": int(k_total),
                    COL_DELTA: np.nan,
                    COL_EPSILON_SINGLE: float(eps),
                    "target_source": target,
                    "selected_historical": json.dumps(selected_hist),
                    "source_values": json.dumps(source_values),
                    "source_roles": json.dumps(source_roles),
                    "t_weights": json.dumps([float(v) for v in t_weights]),
                    COL_RHO_K: "",
                    COL_EPSILON_K: "",
                    "training_data_path": str(training_data_xlsx),
                    "train_count_col": "Count_train",
                    "validation_count_col": "Count_target_val",
                    "test_count_col": "Count_target_test",
                    "reference_name": ref_name,
                    "reference_meta": json.dumps(ref_meta),
                    "shared_precompute_sec": 0.0,
                    "wass_p": float(wass_p),
                }
            )

    radii_df = pd.DataFrame(rows_radii).sort_values(["K_total", "source"], kind="stable")
    train_df_out = pd.DataFrame(rows_train).sort_values(["K_total", COL_ROW_KIND, COL_DELTA, COL_EPSILON_SINGLE], kind="stable")

    radii_csv = run_dir / "radii_by_source.csv"
    params_csv = run_dir / "training_params.csv"
    split_csv = run_dir / "target_split.csv"
    meta_json = run_dir / "meta.json"

    radii_df.to_csv(radii_csv, index=False)
    train_df_out.to_csv(params_csv, index=False)
    target_split = target_pos_df.copy()
    target_split["split"] = "test"
    target_split.loc[target_val_df.index, "split"] = "validation"
    target_split.to_csv(split_csv, index=False)

    meta = {
        "data_path": str(Path(data_path).resolve()),
        "schedule_path": str(Path(schedule_path).resolve()),
        "target_source": target,
        "val_fraction": float(val_fraction),
        "delta_grid": [float(v) for v in delta_grid],
        "epsilon_grid": [float(v) for v in epsilon_grid],
        "wass_p": float(wass_p),
        "affine_static_center": str(affine_static_center),
        "reference_note": "Sampled Gelbrich Gaussian barycenter for W2 radius calibration.",
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "run_dir": run_dir,
        "training_data_xlsx": training_data_xlsx,
        "training_data_csv": training_data_csv,
        "radii_csv": radii_csv,
        "training_params_csv": params_csv,
        "target_split_csv": split_csv,
        "meta_json": meta_json,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compute coffee-shop affine and shared-epsilon radii grids.")
    p.add_argument("--data_path", type=str, required=True, help="Prepared coffee_weekly_groups.xlsx path.")
    p.add_argument("--schedule_path", type=str, required=True, help="coffee_schedule.xlsx path.")
    p.add_argument("--outdir", type=str, required=True, help="Output directory.")
    p.add_argument("--schedule_sheet", type=str, default="schedule", help="Schedule sheet name.")
    p.add_argument("--val_fraction", type=float, default=0.1, help="Target validation fraction.")
    p.add_argument("--delta_grid", type=str, default="[0.0,0.05,0.1,0.2,0.5,1.0]", help="Delta grid for aff/static radii.")
    p.add_argument("--epsilon_grid", type=str, default="[0,0.005,0.01,0.05,0.1,0.2,0.5,1,2,5]", help="Absolute epsilon grid shared by pooled, mixture, minkowski, and barycenter_fitted.")
    p.add_argument("--seed", type=int, default=1234, help="Random seed for target split and W2 barycenter sampling.")
    p.add_argument("--wass_p", type=float, choices=[2.0], default=2.0, help="Wasserstein order p.")
    p.add_argument("--w2_mc_samples", type=int, default=128, help="Number of samples used to discretize the W2 Gaussian barycenter.")
    p.add_argument("--affine_static_center", type=str, default="gelbrich", choices=["gelbrich"],
                   help="Reference center used to set epsilon_k=(1+delta)W(center,P_k) for aff/static when p=2.")
    args = p.parse_args()

    files = run_pipeline(
        data_path=args.data_path,
        schedule_path=args.schedule_path,
        outdir=args.outdir,
        schedule_sheet=args.schedule_sheet,
        val_fraction=float(args.val_fraction),
        delta_grid=parse_float_list(args.delta_grid),
        epsilon_grid=parse_float_list(args.epsilon_grid),
        seed=int(args.seed),
        wass_p=float(args.wass_p),
        w2_mc_samples=int(args.w2_mc_samples),
        affine_static_center=str(args.affine_static_center),
    )
    print(f"Saved under {files['run_dir']}")
    print(f"Training params: {files['training_params_csv']}")


if __name__ == "__main__":
    main()
