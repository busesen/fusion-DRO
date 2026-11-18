"""Data of Section 5.3: download, preprocessing, histogram cells, moment map and the random splits.

Source: Default of Credit Card Clients (UCI Machine Learning Repository, id 350, CC BY 4.0),
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients. The zip file is downloaded once into
--data_dir and checked against a fixed SHA-256. Every preprocessing constant below is fixed in advance and does
not depend on the data.
"""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
ZIP_NAME = "default+of+credit+card+clients.zip"
ZIP_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
XLS_NAME = "default of credit card clients.xls"
N_RECORDS = 30_000

# ------------------------------------------------------------------------------------------------ features
BILL = [f"BILL_AMT{k}" for k in range(1, 7)]
PAYAMT = [f"PAY_AMT{k}" for k in range(1, 7)]
NUM = ["LIMIT_BAL", "AGE"] + BILL + PAYAMT                       # 14 numerical features
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]      # monthly repayment statuses
CAT = ["SEX", "EDUCATION", "MARRIAGE"] + PAY                     # 9 categorical features
LEVELS = [2, 4, 3] + [6] * 6                                      # 45 one-hot inputs
N_NUM, N_CAT, N_ONEHOT = len(NUM), len(CAT), sum(LEVELS)
N_IN = N_NUM + N_ONEHOT                                           # 59 network inputs

# Numerical features: clipped to fixed bounds, transformed, mapped affinely onto [-1, 1].
BOUNDS = {"LIMIT_BAL": (0.0, 1e6, "log1p"), "AGE": (18.0, 100.0, "identity"),
          **{c: (-3.5e5, 1e6, "slog1p") for c in BILL}, **{c: (0.0, 2e6, "log1p") for c in PAYAMT}}
# Fixed bins of the numerical features. The data holder computes the moments from the bin midpoints.
EDGES = {"LIMIT_BAL": [0, 30e3, 50e3, 80e3, 120e3, 180e3, 250e3, 350e3, 500e3, 1e6],
         "AGE": [18, 25, 35, 45, 55, 65, 100],
         **{c: [-3.5e5, 0, 10e3, 25e3, 50e3, 100e3, 200e3, 500e3, 1e6] for c in BILL},
         **{c: [0, 1, 5e3, 10e3, 25e3, 50e3, 100e3, 250e3, 500e3, 2e6] for c in PAYAMT}}

N_CELLS = 432          # label x latest repayment (3) x late months (3) x sex (2) x education (4) x marital status (3)
IU = np.triu_indices(N_NUM)
N_PHI = 2 * N_NUM + len(IU[0]) + 1                                # 134 distinct moments of (x_num, y)


def download(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / ZIP_NAME
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != ZIP_SHA256:
        raise RuntimeError(f"{path} does not match the dataset used in the paper (SHA-256 {ZIP_SHA256})")
    return path


def load_records(data_dir: Path) -> pd.DataFrame:
    """All 30,000 records with the raw UCI codes, sorted by ID; the label is column y."""
    with zipfile.ZipFile(download(data_dir)) as z, z.open(XLS_NAME) as f:
        df = pd.read_excel(f, header=1, engine="xlrd")
    df = df.rename(columns={"default payment next month": "y"})[["ID"] + NUM + CAT + ["y"]].astype(np.int64)
    assert len(df) == N_RECORDS and not df.isna().any().any()
    return df.sort_values("ID").reset_index(drop=True)


def _transform(x, kind):
    x = np.asarray(x, dtype=np.float64)
    return {"log1p": np.log1p, "identity": lambda v: v, "slog1p": lambda v: np.sign(v) * np.log1p(np.abs(v))}[kind](x)


def scale(col: str, x) -> np.ndarray:
    lo, hi, kind = BOUNDS[col]
    t, tlo, thi = _transform(np.clip(x, lo, hi), kind), _transform(lo, kind), _transform(hi, kind)
    return np.clip(2.0 * (t - tlo) / (thi - tlo) - 1.0, -1.0, 1.0)


def bin_midpoints(col: str, x) -> np.ndarray:
    """Scaled midpoint of the fixed bin that contains each value (bins [lower, upper), the last one closed)."""
    lo, hi, kind = BOUNDS[col]
    e = np.asarray(EDGES[col], dtype=np.float64)
    k = np.clip(np.searchsorted(e, np.clip(np.asarray(x, dtype=np.float64), lo, hi), side="right") - 1, 0, len(e) - 2)
    te, tlo, thi = _transform(e, kind), _transform(lo, kind), _transform(hi, kind)
    return (2.0 * (0.5 * (te[:-1] + te[1:]) - tlo) / (thi - tlo) - 1.0)[k]


def categories(df: pd.DataFrame) -> np.ndarray:
    """Raw UCI codes -> category indices. Codes that the UCI documentation does not define are merged:
    EDUCATION 0, 5, 6 -> other (4); MARRIAGE 0 -> other (3); repayment statuses >= 3 -> one level."""
    out = np.empty((len(df), N_CAT), dtype=np.int64)
    out[:, 0] = df["SEX"].to_numpy() - 1
    edu, mar = df["EDUCATION"].to_numpy(), df["MARRIAGE"].to_numpy()
    out[:, 1] = np.select([edu == 1, edu == 2, edu == 3], [0, 1, 2], 3)      # graduate, university, high school, other
    out[:, 2] = np.select([mar == 1, mar == 2], [0, 1], 2)                   # married, single, other
    for k, c in enumerate(PAY):                                              # -2, -1, 0, 1, 2, >= 3
        s = df[c].to_numpy()
        out[:, 3 + k] = np.select([s == -2, s == -1, s == 0, s == 1, s == 2], [0, 1, 2, 3, 4], 5)
    return out


def onehot(cats: np.ndarray) -> np.ndarray:
    out, off = np.zeros((len(cats), N_ONEHOT)), 0
    for k, L in enumerate(LEVELS):
        out[np.arange(len(cats)), off + cats[:, k]] = 1.0
        off += L
    return out


def encode(df: pd.DataFrame) -> dict:
    """Network inputs X = (scaled numerical features, one-hot categories), labels, and the categories and binned
    numerical features from which the data holder computes its statistics."""
    cats = categories(df)
    return {"X": np.concatenate([np.stack([scale(c, df[c]) for c in NUM], 1), onehot(cats)], 1),
            "y": df["y"].to_numpy(), "cats": cats,
            "z_binned": np.stack([bin_midpoints(c, df[c]) for c in NUM], 1)}


def cell(cats: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Histogram cell (0..431) of the label, the latest repayment status PAY_0 (current / one month late / at least
    two months late), the number of late months among PAY_2..PAY_6 (0 / 1-2 / 3-5), sex, education and marital
    status."""
    r = np.where(cats[:, 3] <= 2, 0, np.where(cats[:, 3] == 3, 1, 2))
    late = (cats[:, 4:9] >= 3).sum(1)
    f = np.where(late == 0, 0, np.where(late <= 2, 1, 2))
    return (((9 * y + 3 * r + f) * 2 + cats[:, 0]) * 4 + cats[:, 1]) * 3 + cats[:, 2]


def phi(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(z, upper triangle of z z^T, z y, y): the 134 distinct first and second moments of (x_num, y)."""
    yf = np.asarray(y, dtype=z.dtype)[:, None]
    return np.concatenate([z, z[:, IU[0]] * z[:, IU[1]], z * yf, yf], 1)


# ------------------------------------------------------------------------------------------------ random splits
SPLIT_SEED, ORDER_SEED = 20261601, 20261701
N_POOL, N_TEST = 18_000, 12_000


def _controlled_rounding(rows, cols, frac):
    """0/1 matrix with the given row and column sums (greedy, ties broken by the larger fractional part)."""
    out, cols = np.zeros(frac.shape, dtype=np.int64), cols.copy()
    for s in np.argsort(-rows, kind="stable"):
        for r in sorted(range(frac.shape[1]), key=lambda r: (-cols[r], -frac[s, r]))[:rows[s]]:
            out[s, r] += 1
            cols[r] -= 1
    return out


def split(df: pd.DataFrame, sim: int) -> tuple:
    """Simulation sim: a stratified random split of all records into a pool of 18,000 and 12,000 test records (strata:
    label x latest repayment status, proportional allocation). Returns (pool IDs in random order, test IDs). The N
    hidden records are the first N pool IDs, so they are nested in N."""
    pay0 = df["PAY_0"].to_numpy()
    strat = 3 * df["y"].to_numpy() + np.where(pay0 <= 0, 0, np.where(pay0 == 1, 1, 2))
    labels = np.unique(strat)
    n_s = np.array([(strat == s).sum() for s in labels])
    tot = np.array([N_POOL, N_TEST])
    exp = n_s[:, None] * tot[None, :] / tot.sum()
    base = np.floor(exp).astype(np.int64)
    alloc = base + _controlled_rounding(n_s - base.sum(1), tot - base.sum(0), exp - base)
    rng, ids, pool, test = np.random.default_rng([SPLIT_SEED, sim]), df["ID"].to_numpy(), [], []
    for i, s in enumerate(labels):
        perm = rng.permutation(np.sort(ids[strat == s]))
        pool += perm[:alloc[i, 0]].tolist()
        test += perm[alloc[i, 0]:].tolist()
    pool = np.sort(pool)
    return pool[np.random.default_rng([ORDER_SEED, sim]).permutation(len(pool))], np.sort(test)


def rows(df: pd.DataFrame, ids) -> pd.DataFrame:
    return df[df["ID"].isin(set(ids))].sort_values("ID").reset_index(drop=True)


if __name__ == "__main__":   # download and verify the data, e.g. on a login node with internet access
    print(download(Path(__file__).resolve().parent / "data"))
