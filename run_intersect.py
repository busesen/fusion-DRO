import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from estimators import EstimatorConfig, make_aff_estimator, make_pooled_estimator
from evaluate import evaluate_runs
from simulators import load_excel_bandit
from utils import (
    _ensure_dir,
    _make_aff_initial,
    _make_pooled_initial,
    _parse_estimators,
    _record_aff,
    _record_pooled,
    _resolve_policy_feature_idx,
    _save_run_npz,
    _train,
    parse_float_list,
    parse_str_list,
    project_aff_params,
    project_pooled_params,
    set_seed,
    str2bool,
    validate_prob_vector,
)
from visualization import plot_beta_trajectories, plot_dual_trajectories


def main() -> None:
    p = argparse.ArgumentParser(description="Intersection vs pooled DRO policy learning from Excel")

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--outdir", type=str, default="results")

    p.add_argument("--source_col", type=str, default="hospital")
    p.add_argument("--context_cols", type=str, default="")
    p.add_argument("--treatment_col", type=str, default="treatment")
    p.add_argument("--control_col", type=str, default="control")
    p.add_argument("--train_count_col", type=str, default="COUNT_1")
    p.add_argument("--test_count_col", type=str, default="COUNT_2")
    p.add_argument("--total_count_col", type=str, default="COUNT")
    p.add_argument("--source_values", type=str, default="")
    p.add_argument("--context_scale", type=str, default="zscore", choices=["none", "minmax", "zscore"])
    p.add_argument("--inner_use_unique_contexts", type=str, default="True")

    p.add_argument("--run_aff", type=str, default="True")
    p.add_argument("--run_pooled", type=str, default="True")

    p.add_argument("--epsilons", type=str, required=True, help="List of epsilon_k, one per source")
    p.add_argument("--t_weights", type=str, required=True, help="List of t_k (must sum to 1)")
    p.add_argument("--epsilon_pooled", type=float, default=None)

    p.add_argument("--estimators", type=str, default="saa,mlmc")
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--num_iters", type=int, default=300)
    p.add_argument("--num_runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--n1", type=int, default=32)
    p.add_argument("--saa_inner", type=int, default=128)
    p.add_argument("--mlmc_rate", type=float, default=0.6)
    p.add_argument("--mlmc_truncation", type=int, default=8)

    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--include_linear_mu_eps_term", type=str, default="True")

    p.add_argument("--beta_init", type=str, default="[0.5,0.5]")
    p.add_argument("--aff_mu_init", type=str, default="[1.0]")
    p.add_argument("--pooled_mu_init", type=float, default=1.0)
    p.add_argument("--aff_a_init", type=float, default=0.0)
    p.add_argument("--aff_A_init", type=float, default=0.0)

    p.add_argument("--policy_feature", type=str, default="")
    p.add_argument("--policy_threshold", type=float, default=0.0)

    p.add_argument("--tail", type=int, default=10)

    args = p.parse_args()

    run_aff = str2bool(args.run_aff)
    run_pooled = str2bool(args.run_pooled)
    if not run_aff and not run_pooled:
        raise ValueError("At least one of run_aff/run_pooled must be True")

    context_cols = parse_str_list(args.context_cols) if args.context_cols.strip() else None
    source_values = parse_str_list(args.source_values) if args.source_values.strip() else None

    data = load_excel_bandit(
        data_path=args.data_path,
        source_col=args.source_col,
        context_cols=context_cols,
        treatment_col=args.treatment_col,
        control_col=args.control_col,
        train_count_col=args.train_count_col,
        test_count_col=args.test_count_col,
        total_count_col=args.total_count_col if args.total_count_col.strip() else None,
        source_values=source_values,
        context_scale=args.context_scale,
        use_unique_contexts=str2bool(args.inner_use_unique_contexts),
        dtype=torch.float64,
    )

    epsilons = parse_float_list(args.epsilons)
    t_weights = parse_float_list(args.t_weights)

    if len(epsilons) != data.n_sources:
        raise ValueError(
            f"epsilons length must match number of selected sources K={data.n_sources}, got {len(epsilons)}"
        )
    if len(t_weights) != data.n_sources:
        raise ValueError(
            f"t_weights length must match number of selected sources K={data.n_sources}, got {len(t_weights)}"
        )

    validate_prob_vector(t_weights, "t_weights")

    if args.epsilon_pooled is None:
        epsilon_pool = float(np.dot(np.asarray(t_weights), np.asarray(epsilons)))
    else:
        epsilon_pool = float(args.epsilon_pooled)

    beta_init = parse_float_list(args.beta_init)
    if len(beta_init) != 2:
        raise ValueError("beta_init must contain exactly two values")

    aff_mu_init = parse_float_list(args.aff_mu_init)

    policy_feature_idx = _resolve_policy_feature_idx(data, args.policy_feature)

    estimators = _parse_estimators(args.estimators)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"run_intersect_{ts}")
    files_dir = os.path.join(outdir, "files")
    _ensure_dir(files_dir)

    config = vars(args).copy()
    config["resolved_context_cols"] = data.context_cols
    config["resolved_sources"] = [str(v) for v in data.source_values]
    config["resolved_policy_feature_idx"] = policy_feature_idx
    config["resolved_epsilon_pooled"] = epsilon_pool
    config["K_sources"] = data.n_sources

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    run_records: List[Dict] = []

    for est_name in estimators:
        est_cfg = EstimatorConfig(
            estimator=est_name,
            n1=args.n1,
            saa_inner=args.saa_inner,
            mlmc_rate=args.mlmc_rate,
            mlmc_truncation=args.mlmc_truncation,
            gamma=args.gamma,
            policy_feature_idx=policy_feature_idx,
            policy_threshold=args.policy_threshold,
            include_linear_mu_eps_term=str2bool(args.include_linear_mu_eps_term),
        )

        for run_id in range(args.num_runs):
            run_seed = int(args.seed + 10_000 * (estimators.index(est_name)) + run_id)
            set_seed(run_seed)

            if run_aff:
                x0_aff = _make_aff_initial(
                    beta_init=beta_init,
                    mu_init_raw=aff_mu_init,
                    a_init=args.aff_a_init,
                    a_mat_init=args.aff_A_init,
                    k_sources=data.n_sources,
                    d_ctx=data.d_context,
                    dtype=torch.float64,
                )
                aff_estimator = make_aff_estimator(data, epsilons, t_weights, est_cfg)

                aff_train = _train(
                    x0=x0_aff,
                    num_iters=args.num_iters,
                    learning_rate=args.learning_rate,
                    optimizer_name=args.optimizer,
                    estimator_fn=aff_estimator,
                    project_fn=lambda x: project_aff_params(x, data.n_sources, data.d_context),
                    n1=args.n1,
                )

                aff_rec = _record_aff(
                    aff_train,
                    meta={
                        "method": "aff",
                        "estimator": est_name,
                        "optimizer": args.optimizer,
                        "run_id": run_id,
                        "seed": run_seed,
                    },
                    k_sources=data.n_sources,
                    d_ctx=data.d_context,
                )
                run_records.append(aff_rec)

                npz_name = f"aff_{est_name}_{args.optimizer}_run{run_id}.npz"
                _save_run_npz(os.path.join(files_dir, npz_name), aff_rec)

            if run_pooled:
                x0_pool = _make_pooled_initial(beta_init, args.pooled_mu_init, dtype=torch.float64)
                pooled_estimator = make_pooled_estimator(data, epsilon_pool, est_cfg)

                pooled_train = _train(
                    x0=x0_pool,
                    num_iters=args.num_iters,
                    learning_rate=args.learning_rate,
                    optimizer_name=args.optimizer,
                    estimator_fn=pooled_estimator,
                    project_fn=project_pooled_params,
                    n1=args.n1,
                )

                pooled_rec = _record_pooled(
                    pooled_train,
                    meta={
                        "method": "pooled",
                        "estimator": est_name,
                        "optimizer": args.optimizer,
                        "run_id": run_id,
                        "seed": run_seed,
                    },
                )
                run_records.append(pooled_rec)

                npz_name = f"pooled_{est_name}_{args.optimizer}_run{run_id}.npz"
                _save_run_npz(os.path.join(files_dir, npz_name), pooled_rec)

    # Save compact metadata for all runs.
    meta_runs = []
    for r in run_records:
        meta_runs.append(
            {
                "method": r["method"],
                "estimator": r["estimator"],
                "optimizer": r["optimizer"],
                "run_id": r["run_id"],
                "seed": r["seed"],
            }
        )
    with open(os.path.join(outdir, "runs.json"), "w") as f:
        json.dump(meta_runs, f, indent=2)

    eval_df = evaluate_runs(
        run_records=run_records,
        data=data,
        tail=args.tail,
        feature_idx=policy_feature_idx,
        threshold=args.policy_threshold,
    )
    eval_csv = os.path.join(outdir, f"eval_tail{args.tail}.csv")
    eval_df.to_csv(eval_csv, index=False)

    if not eval_df.empty:
        summary_df = (
            eval_df.groupby(["method", "estimator", "optimizer"], as_index=False)
            .agg(
                num_runs=("run_id", "count"),
                test_cost_mean=("test_expected_cost", "mean"),
                test_cost_std=("test_expected_cost", "std"),
                train_cost_mean=("train_expected_cost", "mean"),
                train_cost_std=("train_expected_cost", "std"),
            )
            .sort_values(["method", "estimator", "optimizer"])
        )
    else:
        summary_df = pd.DataFrame()

    summary_csv = os.path.join(outdir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    plot_beta_trajectories(run_records, os.path.join(outdir, "beta_iterates.png"), title="Policy Parameters")
    plot_dual_trajectories(run_records, os.path.join(outdir, "dual_iterates.png"), title="Dual Variables")

    print(f"Completed. Results written to: {outdir}")
    print(f"Evaluation CSV: {eval_csv}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
