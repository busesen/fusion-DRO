"""Coffee sales data and the multi-item newsvendor cost of Section 5.2."""
from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RAW_COFFEE_XLSX = Path(__file__).resolve().parent / "data" / "Coffee_Shop_Sales.xlsx"
DEMAND_COLS = ["Sm", "Rg", "Lg"]          # small, regular and large cups
PRODUCT_NAMES = {"group_1": "Brazilian", "group_2": "Colombian", "group_3": "Ethiopian",
                 "group_4": "Jamaican", "group_5": "Old Time"}

# Cost coefficients: unit prices p are the modal Brazilian prices (first product); the order
# costs c and the holding costs h = 0.05 p follow Li et al. (2026); backorder costs b = 1.5 p.
ORDER_COST = np.array([0.4, 1.9, 2.0], dtype=float)
HOLDING_MULT = 0.05
BACKORDER_MULT = 1.5


def parse_list(raw: object) -> List[str]:
    s = str(raw).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        return [str(v) for v in json.loads(s)]
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def parse_float_list(raw: object) -> List[float]:
    return [float(v) for v in parse_list(raw)]


def _mode_price(x: pd.Series) -> float:
    m = x.mode(dropna=True)
    return float(m.iloc[0]) if len(m) else float("nan")


def load_weekly_demand(raw_xlsx: str | Path = RAW_COFFEE_XLSX) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Weekly average daily demand of the coffee products sold in all three cup sizes.

    Returns {product id: array of shape (weeks, 3)}, summed over the three stores, and the
    unit prices (modal price per cup size) of the first product."""
    df = pd.read_excel(raw_xlsx)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df = df[df["product_category"].eq("Coffee")].copy()
    parts = df["product_detail"].astype(str).str.extract(r"^(?P<base>.*)\s+(?P<size>Sm|Rg|Lg)$")
    df["base"], df["size"] = parts["base"], parts["size"]
    df = df.dropna(subset=["base", "size"])
    all_sizes = df.groupby("base")["size"].apply(lambda s: set(DEMAND_COLS).issubset(set(s)))
    df = df[df["base"].isin(all_sizes[all_sizes].index)].copy()

    df["date"] = df["transaction_date"].dt.date
    daily = df.groupby(["base", "size", "date"], as_index=False)["transaction_qty"].sum()
    daily["week"] = pd.to_datetime(daily["date"]).dt.to_period("W-MON").apply(lambda p: p.start_time.date())
    weekly = daily.groupby(["base", "size", "week"], as_index=False)["transaction_qty"].mean()

    demand: Dict[str, np.ndarray] = {}
    for idx, coffee in enumerate(sorted(weekly["base"].unique()), start=1):
        one = (weekly[weekly["base"] == coffee]
               .pivot_table(index="week", columns="size", values="transaction_qty", aggfunc="mean", fill_value=0)
               .reindex(columns=DEMAND_COLS, fill_value=0)
               .sort_index())
        demand[f"group_{idx}"] = one.to_numpy(dtype=float)

    prices = (df.groupby(["base", "size"])["unit_price"].agg(_mode_price).unstack("size")
              .reindex(columns=DEMAND_COLS).sort_index())
    return demand, prices.iloc[0].to_numpy(dtype=float)


def cost_parameters(prices: np.ndarray) -> Dict[str, np.ndarray]:
    P = np.asarray(prices, dtype=float)
    return {"P": P, "C": ORDER_COST.copy(), "H": HOLDING_MULT * P, "B": BACKORDER_MULT * P, "dim": len(P)}


def branch_matrices(cost_params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Affine pieces of the multi-item newsvendor loss.

    The loss is a sum over items of a per-item maximum,
        l(w; xi) = sum_i max{(C_i - P_i - B_i) w_i + B_i xi_i,  (C_i + H_i) w_i - (H_i + P_i) xi_i},
    so as a single pointwise maximum it has 2^d pieces, one per choice of understocked or
    overstocked for every item: l(w; xi) = max_l a_x[l] @ w + a_xi[l] @ xi.
    """
    H, B, C, P = (np.asarray(cost_params[k], dtype=float) for k in ("H", "B", "C", "P"))
    under_x, under_xi = C - P - B, B
    over_x, over_xi = C + H, -H - P
    rows_x, rows_xi = [], []
    for pattern in itertools.product((0, 1), repeat=C.shape[0]):
        mask = np.asarray(pattern, dtype=bool)        # True = item overstocked
        rows_x.append(np.where(mask, over_x, under_x))
        rows_xi.append(np.where(mask, over_xi, under_xi))
    return np.asarray(rows_x, dtype=float), np.asarray(rows_xi, dtype=float)


def newsvendor_cost_matrix(demand: np.ndarray, w: np.ndarray, cost_params: Dict[str, np.ndarray]) -> np.ndarray:
    """Newsvendor cost of the order quantities w for every demand row (negative = profit)."""
    x = np.asarray(demand, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    H, B, C, P = (np.asarray(cost_params[k], dtype=float) for k in ("H", "B", "C", "P"))
    sold = np.minimum(w[None, :], x)
    over = np.maximum(w[None, :] - x, 0.0)
    under = np.maximum(x - w[None, :], 0.0)
    return np.asarray(-sold @ P + np.sum(w * C) + over @ H + under @ B, dtype=float)
