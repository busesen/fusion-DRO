from __future__ import annotations

import itertools
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT_ENV = "IWB_REPO_ROOT"


def _resolve_root_dir() -> Path:
    raw = os.environ.get(REPO_ROOT_ENV, "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        if p.name == "coffee-newsvendor":
            return p.parent.resolve()
        return p
    return Path(__file__).resolve().parent.parent


ROOT_DIR = _resolve_root_dir()
COFFEE_DIR = ROOT_DIR / "coffee-newsvendor"
RAW_COFFEE_XLSX = COFFEE_DIR / "data" / "Coffee_Shop_Sales.xlsx"

GROUP_COL = "group_id"
BASE_COL = "base_coffee"
WEEK_COL = "week"
COUNT_COL = "Count"
DEMAND_COLS = ["Sm", "Rg", "Lg"]

# These are exactly the coefficients used in RooD-SO's coffee setup.
DEFAULT_ORDER_COST = np.array([0.4, 1.9, 2.0], dtype=float)


def str2bool(x: object) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_list(raw: object) -> List[str]:
    if isinstance(raw, list):
        return [str(v) for v in raw]
    s = str(raw).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def parse_float_list(raw: object) -> List[float]:
    vals = parse_list(raw)
    return [float(v) for v in vals]


def mode_price(x: pd.Series) -> float:
    m = x.mode(dropna=True)
    return float(m.iloc[0]) if len(m) else float("nan")


def _extract_base_and_size(df: pd.DataFrame) -> pd.DataFrame:
    pat = re.compile(r"^(?P<base>.*)\s+(?P<size>Sm|Rg|Lg)$")
    tmp = df["product_detail"].astype(str).str.extract(pat)
    out = df.copy()
    out[BASE_COL] = tmp["base"]
    out["size"] = tmp["size"]
    out = out.dropna(subset=[BASE_COL, "size"]).copy()
    return out


def build_weekly_group_data(raw_xlsx: str | Path = RAW_COFFEE_XLSX) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    df = pd.read_excel(raw_xlsx).copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    if "product_category" in df.columns:
        df = df[df["product_category"].eq("Coffee")].copy()
    df = _extract_base_and_size(df)

    sizes_required = {"Sm", "Rg", "Lg"}
    valid_base = (
        df.groupby(BASE_COL)["size"]
        .apply(lambda s: sizes_required.issubset(set(s)))
    )
    valid_base = valid_base[valid_base].index.tolist()
    df = df[df[BASE_COL].isin(valid_base)].copy()

    df["date"] = df["transaction_date"].dt.date
    daily = (
        df.groupby([BASE_COL, "size", "date"], as_index=False)["transaction_qty"]
        .sum()
    )
    daily[WEEK_COL] = pd.to_datetime(daily["date"]).dt.to_period("W-MON").apply(lambda p: p.start_time.date())
    weekly_mean = (
        daily.groupby([BASE_COL, "size", WEEK_COL], as_index=False)["transaction_qty"]
        .mean()
        .rename(columns={"transaction_qty": "avg_daily_cups"})
    )

    coffee_list = sorted(weekly_mean[BASE_COL].unique().tolist())
    group_frames: List[pd.DataFrame] = []
    for idx, coffee in enumerate(coffee_list, start=1):
        one = weekly_mean[weekly_mean[BASE_COL] == coffee].copy()
        one = one.pivot_table(
            index=WEEK_COL,
            columns="size",
            values="avg_daily_cups",
            aggfunc="mean",
            fill_value=0,
        ).reindex(columns=DEMAND_COLS, fill_value=0)
        one = one.reset_index()
        one[GROUP_COL] = f"group_{idx}"
        one[BASE_COL] = coffee
        one = one[[GROUP_COL, BASE_COL, WEEK_COL] + DEMAND_COLS].copy()
        group_frames.append(one)

    price_long = (
        df.groupby([BASE_COL, "size"])["unit_price"]
        .agg(unit_price=mode_price, nunique="nunique")
        .reset_index()
    )
    price_pivot = (
        price_long.pivot(index=BASE_COL, columns="size", values="unit_price")
        .reindex(columns=DEMAND_COLS)
        .reset_index()
    )
    return group_frames, price_pivot


def make_long_weekly_frame(group_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for g in group_frames:
        one = g.copy()
        one[COUNT_COL] = 1.0
        rows.append(one)
    out = pd.concat(rows, axis=0, ignore_index=True)
    out[GROUP_COL] = out[GROUP_COL].astype(str)
    out[BASE_COL] = out[BASE_COL].astype(str)
    out[COUNT_COL] = pd.to_numeric(out[COUNT_COL], errors="coerce").fillna(0.0)
    for col in DEMAND_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out[[GROUP_COL, BASE_COL, WEEK_COL] + DEMAND_COLS + [COUNT_COL]].copy()


def load_cost_params(price_csv: str | Path) -> Dict[str, np.ndarray]:
    price_df = pd.read_csv(price_csv)
    if price_df.empty:
        raise ValueError(f"Price CSV is empty: {price_csv}")
    # RooD-SO uses the first row of the price table for the coffee experiment.
    price_vec = price_df.loc[0, DEMAND_COLS].to_numpy(dtype=float)
    P = np.asarray(price_vec, dtype=float)
    C = DEFAULT_ORDER_COST.copy()
    H = 0.05 * P
    B = 0.5 * P
    return {
        "H": H,
        "B": B,
        "C": C,
        "P": P,
        "dim": int(len(DEMAND_COLS)),
    }


def branch_matrices(cost_params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Affine pieces of the multi-item newsvendor loss.

    The loss is a sum over items of a per-item maximum,
        l(w; xi) = sum_i max{(C_i - P_i - B_i) w_i + B_i xi_i,  (C_i + H_i) w_i - (H_i + P_i) xi_i},
    so as a single pointwise maximum it has 2^d pieces, one per choice of
    understocked/overstocked for every item. Row l of (a_x, a_xi) is one piece:
    l(w; xi) = max_l a_x[l] @ w + a_xi[l] @ xi, which equals newsvendor_cost_matrix.
    """
    H = np.asarray(cost_params["H"], dtype=float)
    B = np.asarray(cost_params["B"], dtype=float)
    C = np.asarray(cost_params["C"], dtype=float)
    P = np.asarray(cost_params["P"], dtype=float)
    d = int(C.shape[0])
    under_x, under_xi = C - P - B, B
    over_x, over_xi = C + H, -H - P
    rows_x: List[np.ndarray] = []
    rows_xi: List[np.ndarray] = []
    for pattern in itertools.product((0, 1), repeat=d):
        mask = np.asarray(pattern, dtype=bool)  # True = item overstocked
        rows_x.append(np.where(mask, over_x, under_x))
        rows_xi.append(np.where(mask, over_xi, under_xi))
    return np.asarray(rows_x, dtype=float), np.asarray(rows_xi, dtype=float)


def newsvendor_cost_matrix(demand: np.ndarray, w: np.ndarray, cost_params: Dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(demand, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    H = np.asarray(cost_params["H"], dtype=float)
    B = np.asarray(cost_params["B"], dtype=float)
    C = np.asarray(cost_params["C"], dtype=float)
    P = np.asarray(cost_params["P"], dtype=float)
    sold = np.minimum(w[None, :], x)
    over = np.maximum(w[None, :] - x, 0.0)
    under = np.maximum(x - w[None, :], 0.0)
    cost = -sold @ P + np.sum(w * C) + over @ H + under @ B
    return np.asarray(cost, dtype=float)


def expected_cost_from_counts(df: pd.DataFrame, w: Sequence[float], cost_params: Dict[str, np.ndarray], count_col: str) -> float:
    counts = pd.to_numeric(df[count_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mask = counts > 0.0
    if not np.any(mask):
        return float("nan")
    demand = df.loc[mask, DEMAND_COLS].to_numpy(dtype=float)
    costs = newsvendor_cost_matrix(demand, np.asarray(w, dtype=float), cost_params)
    return float(np.dot(costs, counts[mask]) / counts[mask].sum())


def source_sample_list(df: pd.DataFrame, source_values: Sequence[str], demand_cols: Sequence[str] = DEMAND_COLS, count_col: str = COUNT_COL) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for source in source_values:
        part = df[df[GROUP_COL].astype(str) == str(source)].copy()
        counts = pd.to_numeric(part[count_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mask = counts > 0.0
        samples = part.loc[mask, list(demand_cols)].to_numpy(dtype=float)
        if samples.shape[0] == 0:
            raise ValueError(f"No positive-mass rows for source {source}")
        out.append(samples)
    return out


def uniform_probs(n: int) -> np.ndarray:
    return np.ones(int(n), dtype=float) / float(int(n))


def weighted_mixture_support(samples_list: Sequence[np.ndarray], weights: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    samples = np.vstack([np.asarray(s, dtype=float) for s in samples_list])
    probs = np.concatenate([
        np.full(s.shape[0], weights[k] / float(s.shape[0]), dtype=float)
        for k, s in enumerate(samples_list)
    ])
    probs = probs / probs.sum()
    return samples, probs


def pooled_support(samples_list: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    samples = np.vstack([np.asarray(s, dtype=float) for s in samples_list])
    return samples, uniform_probs(samples.shape[0])
