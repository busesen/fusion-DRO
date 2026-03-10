from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

from simulators import ExcelBanditData
from utils import expected_cost, weighted_average


def evaluate_beta_on_split(
    data: ExcelBanditData,
    beta: torch.Tensor,
    feature_idx: int,
    threshold: float,
    split: str = "test",
) -> float:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be train/test, got {split}")

    weights = data.train_counts if split == "train" else data.test_counts
    costs = expected_cost(
        contexts=data.contexts,
        treatment=data.treatment,
        control=data.control,
        beta=beta,
        feature_idx=feature_idx,
        threshold=threshold,
    )
    val = weighted_average(costs, weights)
    return float(val.detach().cpu().item())


def evaluate_beta_by_source(
    data: ExcelBanditData,
    beta: torch.Tensor,
    feature_idx: int,
    threshold: float,
    split: str = "test",
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for s_idx, s_val in enumerate(data.source_values):
        idx = data.source_indices[s_idx]
        if split == "train":
            w = data.train_counts[idx]
        else:
            w = data.test_counts[idx]
        c = expected_cost(
            contexts=data.contexts[idx],
            treatment=data.treatment[idx],
            control=data.control[idx],
            beta=beta,
            feature_idx=feature_idx,
            threshold=threshold,
        )
        out[str(s_val)] = float(weighted_average(c, w).detach().cpu().item())
    return out


def _tail_avg_beta(beta_traj: np.ndarray, tail: int) -> np.ndarray:
    if beta_traj.ndim != 2 or beta_traj.shape[1] != 2:
        raise ValueError("beta_traj must have shape (T, 2)")
    t = min(tail, beta_traj.shape[0])
    return beta_traj[-t:].mean(axis=0)


def evaluate_runs(
    run_records: Iterable[Dict],
    data: ExcelBanditData,
    tail: int,
    feature_idx: int,
    threshold: float,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for rec in run_records:
        beta_tail_np = _tail_avg_beta(np.asarray(rec["beta"], dtype=float), tail)
        beta_tail = torch.as_tensor(beta_tail_np, dtype=data.contexts.dtype)

        row = {
            "method": rec.get("method", ""),
            "estimator": rec.get("estimator", ""),
            "optimizer": rec.get("optimizer", ""),
            "run_id": rec.get("run_id", -1),
            "tail": int(tail),
            "beta1_tail_avg": float(beta_tail_np[0]),
            "beta2_tail_avg": float(beta_tail_np[1]),
            "train_expected_cost": evaluate_beta_on_split(
                data=data,
                beta=beta_tail,
                feature_idx=feature_idx,
                threshold=threshold,
                split="train",
            ),
            "test_expected_cost": evaluate_beta_on_split(
                data=data,
                beta=beta_tail,
                feature_idx=feature_idx,
                threshold=threshold,
                split="test",
            ),
        }

        mu_arr = np.asarray(rec.get("mu", []), dtype=float)
        if mu_arr.size > 0:
            t = min(tail, mu_arr.shape[0])
            row["mu_tail_avg"] = float(mu_arr[-t:].mean())

        if "mu_k" in rec:
            mu_k = np.asarray(rec["mu_k"], dtype=float)
            if mu_k.ndim == 2 and mu_k.size > 0:
                t = min(tail, mu_k.shape[0])
                mu_k_tail = mu_k[-t:].mean(axis=0)
                row["mu_k_tail_avg_mean"] = float(mu_k_tail.mean())
                row["mu_k_tail_avg_min"] = float(mu_k_tail.min())
                row["mu_k_tail_avg_max"] = float(mu_k_tail.max())

        rows.append(row)

    return pd.DataFrame(rows)
