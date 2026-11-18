"""Section 5.3: neural network training from noisy moments and a noisy histogram.

One call handles one number N of hidden records, a list of simulations and a list of methods:

    python credit-classification/run_classification.py --n 6000 --sims 0-9 --methods affine,static

For every simulation it draws the split, computes the releases of the N hidden records and certifies that the
fusion ambiguity set is nonempty (cached in <outdir>/releases), then trains every method on all listed simulations
at once (one network per simulation) and writes the test cross-entropies to <outdir>/n<N>/<method>_sims<a>-<b>.csv.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import credit_data as D
import releases as R
import sgd
from methods import Problem

HERE = Path(__file__).resolve().parent
METHODS = ["affine", "static", "minkowski", "moment", "tv", "oracle"]
INIT_SEED, SEARCH_SEED = 20261501, 20261801
N_CANDIDATES = 2_000   # candidate records of the nonemptiness certificate: the records that are hidden for every N


def parse_sims(s: str) -> list:
    out = []
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def get_release(df, pool, sim: int, n: int, outdir: Path) -> dict:
    """releases of the first n pool records, computed and certified once and then read from the cache"""
    f = outdir / "releases" / f"n{n}" / f"sim{sim:02d}.npz"
    if not f.exists():
        rel = R.release(D.encode(D.rows(df, pool[:n])), sim)
        cert = R.certify_nonempty(rel, D.encode(D.rows(df, pool[:N_CANDIDATES])))
        if not cert["certified"]:
            raise RuntimeError(f"n={n}, simulation {sim}: nonempty intersection not certified ({cert})")
        print(f"n={n} sim {sim}: r_M {rel['r_M']:.4f}, r_TV {rel['r_TV']:.4f}, certified nonempty by a distribution "
              f"on {cert['support_size']} points", flush=True)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f"{f.stem}_{os.getpid()}.npz")
        np.savez(tmp, **rel)
        os.replace(tmp, f)
    return dict(np.load(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="number of hidden records N (paper: 2000, 6000, ..., 18000)")
    ap.add_argument("--sims", default="0-49", help="simulations, e.g. 0-9 or 0,3,5 (paper: 0-49)")
    ap.add_argument("--methods", default=",".join(METHODS), help=f"comma-separated subset of {','.join(METHODS)}")
    ap.add_argument("--steps", type=int, default=75_000, help="updates of the robust methods")
    ap.add_argument("--outdir", type=Path, default=HERE / "results")
    ap.add_argument("--data_dir", type=Path, default=HERE / "data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--releases_only", action="store_true", help="only compute and certify the releases")
    ap.add_argument("--no_cuda_graph", action="store_true", help="run the inner maximizations without a CUDA graph")
    a = ap.parse_args()
    sims, methods, dev = parse_sims(a.sims), a.methods.split(","), torch.device(a.device)
    assert set(methods) <= set(METHODS), f"choose methods from {METHODS}"

    df = D.load_records(a.data_dir)
    splits = {s: D.split(df, s) for s in sims}
    rel = {s: get_release(df, splits[s][0], s, a.n, a.outdir) for s in sims}
    if a.releases_only:
        return
    f32 = lambda arrays: torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=dev)
    tests = [D.encode(D.rows(df, splits[s][1])) for s in sims]
    Xt, yt = f32([T["X"] for T in tests]), torch.as_tensor(np.stack([T["y"] for T in tests]), device=dev)
    for method in methods:
        t0 = time.time()
        net = sgd.networks([INIT_SEED + s for s in sims], dev)
        row = {"n": a.n, "method": method, "sim": sims}
        if method == "oracle":
            hidden = [D.encode(D.rows(df, splits[s][0][:a.n])) for s in sims]
            X, y = f32([h["X"] for h in hidden]), torch.as_tensor(np.stack([h["y"] for h in hidden]), device=dev)
            ce, step = sgd.train_oracle(net, X, y, [INIT_SEED * 100 + 37 * s for s in sims], Xt, yt)
            row |= {"test_ce": ce.tolist(), "update": step.tolist()}
        else:
            stacked = {k: f32([rel[s][k] for s in sims]) for k in ("m_hat", "r_M", "p_hat", "r_TV")}
            seed = np.random.SeedSequence([SEARCH_SEED, a.n, METHODS.index(method), *sims]).generate_state(1)[0]
            P = Problem(method, stacked, seed, cuda_graph=dev.type == "cuda" and not a.no_cuda_graph)
            sgd.train_robust(P, net, a.steps)
            row |= {"test_ce": sgd.test_ce(net, Xt, yt).tolist(), "update": a.steps}
        row["seconds"] = time.time() - t0
        out = a.outdir / f"n{a.n}" / f"{method}_sims{sims[0]}-{sims[-1]}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(row).to_csv(out, index=False)
        print(f"{method}, n={a.n}: mean test cross-entropy {np.mean(row['test_ce']):.4f} "
              f"({row['seconds']:.0f} s) -> {out}", flush=True)


if __name__ == "__main__":
    main()
