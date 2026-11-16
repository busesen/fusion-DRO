"""Section 5.2: multi-item newsvendor on the Maven Roasters coffee sales data.

Each of the five coffee products serves in turn as the target. In every repetition (split),
`--n_known` target weeks are drawn at random and the remaining weeks form the test set. The
full histories of the four other products are corrupted by bulk orders: each of their weeks
is multiplied by 4 with probability 10%, 20%, 30% or 40%, assigned to the four products in
random order. The target weeks are never corrupted. The K = 5 reference distributions are the
four corrupted histories and the known target weeks, with weights 1/8 and 1/2.

Hyperparameters are selected by leave-one-out cross-validation over the known target weeks:
in each fold, one known week is the validation week and the others form the target
reference. Each method is then re-solved with all known weeks at the hyperparameter with the
lowest average validation cost (the smallest one in case of ties) and evaluated on the test
weeks. Only hyperparameters that solve to optimality in every fold are eligible.

  affine, static         eps_k = (1 + delta) * W2(P_k, P_bary), delta on --delta_grid
  pool, mixture,         one radius eps on --epsilon_grid (minkowski: eps_k = eps for all k)
  minkowski, barycenter
  erm                    no hyperparameter
  rkhs                   nu on --nu_grid (Li et al., 2026)

P_bary is the empirical distribution of --barycenter_num_samples samples from the Gelbrich
barycenter of the references. The same sample is the center of barycenter and anchors the
radii of affine and static.

Every random quantity has its own seed, derived from (--seed, target, split, fold, purpose),
so the results do not depend on the number of workers or the order of execution. MOSEK and
Gurobi run with a fixed number of threads. The reported time of a method is the wall-clock
time of its final solve at the selected hyperparameter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# One BLAS/OpenMP thread per worker (before numpy is imported): the workers already run in parallel.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# MOSEK and Gurobi read their license location when they are loaded.
for _i, _a in enumerate(sys.argv):
    for _flag, _var in {"--mosek_license_file": "MOSEKLM_LICENSE_FILE", "--gurobi_license_file": "GRB_LICENSE_FILE"}.items():
        if _a == _flag and _i + 1 < len(sys.argv) and sys.argv[_i + 1].strip():
            os.environ[_var] = os.path.expanduser(sys.argv[_i + 1])

import numpy as np
import pandas as pd

import rkhs
import solvers
from coffee_data import RAW_COFFEE_XLSX, cost_parameters, load_weekly_demand, newsvendor_cost_matrix, parse_float_list, parse_list

METHODS = ["erm", "pool", "mixture", "minkowski", "barycenter", "static", "static_sq", "affine", "rkhs"]
CONIC_METHODS = METHODS[:-1]
BULK_PROBABILITIES = (0.10, 0.20, 0.30, 0.40)
BULK_FACTOR = 4.0
DELTA_GRID = "[0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]"
NU_GRID = "[" + ", ".join(f"{v:.2f}" for v in np.arange(1, 20) * 0.05) + "]"
PURPOSE = {"split": 1, "corrupt": 2, "bary": 4, "rkhs": 5, "perm": 6}
STREAM_KEY = (4, 2000)    # fixed integers that, with --seed, identify the random streams of this experiment


def sub_seed(base: int, *keys: int) -> int:
    """Deterministic 32-bit seed from integer keys."""
    return int(np.random.SeedSequence([int(base)] + [int(k) for k in keys]).generate_state(1)[0])


def corrupt(X: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Bulk orders: every week is multiplied by BULK_FACTOR with probability BULK_PROBABILITIES[level]."""
    X = np.asarray(X, dtype=float).copy()
    X[rng.random(len(X)) < BULK_PROBABILITIES[level]] *= BULK_FACTOR
    return X


def make_instance(W: Dict[str, np.ndarray], target: str, split: int, seed: int, n_known: int, target_weight: float):
    """Known and test weeks of the target, corrupted histories of the other products, and weights."""
    groups = sorted(W)
    gi = groups.index(target)
    idx = np.random.default_rng(sub_seed(seed, PURPOSE["split"], gi, split)).permutation(len(W[target]))
    known, test = idx[:n_known], idx[n_known:]
    others = [h for h in groups if h != target]
    levels = np.random.default_rng(sub_seed(seed, PURPOSE["perm"], gi, split)).permutation(len(others))
    rng = np.random.default_rng(sub_seed(seed, PURPOSE["corrupt"], gi, split, *STREAM_KEY))
    hist = [corrupt(W[h], int(levels[j]), rng) for j, h in enumerate(others)]
    t = np.array([(1.0 - target_weight) / len(others)] * len(others) + [target_weight])
    return known, test, others, hist, t


def cost(X: np.ndarray, w: np.ndarray, cp_: Dict[str, Any]) -> float:
    return float(newsvendor_cost_matrix(np.atleast_2d(X), np.asarray(w, dtype=float), cp_).mean())


def is_ok(status: str) -> bool:
    return str(status).startswith("optimal")


def solve_all(refs: List[np.ndarray], t: np.ndarray, cp_: Dict[str, Any], args, seeds: Dict[str, Any],
              only: Dict[str, float] | None = None) -> List[Dict[str, Any]]:
    """Solve every (method, hyperparameter) once for the given references; with `only`, solve
    just the given method -> hyperparameter pairs."""
    solver = args.cvxpy_solver
    t = np.asarray(t, float) / np.sum(t)
    pooled = np.vstack(refs)
    pooled_w = np.full(len(pooled), 1.0 / len(pooled))
    mix_w = np.concatenate([np.full(len(X), t[k] / len(X)) for k, X in enumerate(refs)])
    uniform = [np.full(len(X), 1.0 / len(X)) for X in refs]
    # one sample from the Gelbrich barycenter is the center of barycenter and anchors the radii
    # eps_k = (1 + delta) * W2(P_k, P_bary) of affine and static
    bary, bary_w = solvers.gaussian_barycenter_samples(refs, t, args.barycenter_num_samples, seeds["bary"])
    rho = np.array([solvers.w2_distance(bary, bary_w, X, u) for X, u in zip(refs, uniform)])
    out: List[Dict[str, Any]] = []
    clock = {"t": float("nan"), "solver": ""}

    def want(method, h):
        return method in args.methods and (only is None or (method in only and float(only[method]) == float(h)))

    def timed(fn, *fargs):
        t0 = time.perf_counter()
        res = fn(*fargs)
        clock["t"], clock["solver"] = time.perf_counter() - t0, solvers.LAST_SOLVER_USED["name"]
        return res

    def rec(method, h, res):
        st, obj, w, aux = res
        out.append(dict(method=method, h=float(h), status=str(st), objective=obj, solve_time=clock["t"],
                        solver=clock["solver"], w=None if w is None else np.asarray(w, float).tolist(), aux=aux or {}))

    if want("erm", 0.0):
        rec("erm", 0.0, timed(solvers.solve_single_ball, pooled, pooled_w, 0.0, cp_, solver))
    for e in args.epsilon_grid:
        if want("pool", e):
            rec("pool", e, timed(solvers.solve_single_ball, pooled, pooled_w, e ** 2, cp_, solver))
        if want("mixture", e):
            rec("mixture", e, timed(solvers.solve_single_ball, pooled, mix_w, e ** 2, cp_, solver))
        if want("minkowski", e):
            rec("minkowski", e, timed(solvers.solve_minkowski, refs, t, np.full(len(refs), e ** 2), cp_, solver))
        if want("barycenter", e):
            rec("barycenter", e, timed(solvers.solve_single_ball, bary, bary_w, e ** 2, cp_, solver))
    for dl in args.delta_grid:
        eps_k = (1.0 + dl) * rho
        if want("static", dl):
            rec("static", dl, timed(solvers.solve_static, refs, t, eps_k ** 2, cp_, solver))
        if want("static_sq", dl):
            rec("static_sq", dl, timed(solvers.solve_static_sq, refs, t, eps_k ** 2, cp_, solver))
        if want("affine", dl):
            rec("affine", dl, timed(solvers.solve_affine, refs, t, eps_k ** 2, cp_, solver))
    for nu in args.nu_grid:
        if want("rkhs", nu):
            t0 = time.perf_counter()
            res = rkhs.solve_rkhs(refs, refs[-1], cp_, seeds["rkhs_outdir"], verbose=False, seed=seeds["rkhs"],
                                  rkhs_nu=float(nu), rkhs_source_weights=np.ones(len(refs)) / len(refs),
                                  rkhs_source_weight_mode="uniform", rkhs_n_J=args.rkhs_n_J, rkhs_is_herding=False,
                                  rkhs_T=args.rkhs_T, rkhs_time_limit=args.rkhs_time_limit,
                                  gurobi_license_file=args.gurobi_license_file)
            clock["t"], clock["solver"] = time.perf_counter() - t0, "GUROBI"
            rec("rkhs", nu, res)
    for r in out:
        r["rho"] = rho.tolist()
    return out


def _warm_up(args, cp_: Dict[str, Any]) -> None:
    """Solve a tiny problem once per worker before anything is timed, so that one-time costs
    (loading the solver, license check) are not attributed to the first method."""
    if all(m == "rkhs" for m in args.methods):
        return
    X = np.abs(np.random.default_rng(0).normal(10.0, 2.0, size=(4, int(cp_["dim"]))))
    try:
        solvers.solve_single_ball(X, np.full(4, 0.25), 0.1, cp_, args.cvxpy_solver)
        solvers.solve_affine([X, X + 1.0], np.array([0.5, 0.5]), np.array([0.1, 0.1]), cp_, args.cvxpy_solver)
    except Exception:  # noqa: BLE001
        pass


def run_instance(job: Dict[str, Any], args_dict: Dict[str, Any]) -> str:
    args = argparse.Namespace(**args_dict)
    solvers.SOLVER_OPTS.clear()
    rkhs.GUROBI_PARAMS.clear()
    if args.gurobi_threads > 0:
        rkhs.GUROBI_PARAMS["Threads"] = int(args.gurobi_threads)
    if args.cvxpy_solver.upper() == "MOSEK" and args.solver_threads > 0:
        solvers.SOLVER_OPTS["mosek_params"] = {"MSK_IPAR_NUM_THREADS": int(args.solver_threads)}
    outdir = Path(job["outdir"])
    done_flag = outdir / f"selected_{args.tag}.csv"
    if done_flag.exists():
        return f"skip {outdir.name}"
    outdir.mkdir(parents=True, exist_ok=True)
    _warm_up(args, job["cost_params"])

    W = {g: np.asarray(v, float) for g, v in job["W"].items()}
    g, split, cp_, base = job["target"], job["split"], job["cost_params"], args.seed
    gi = sorted(W).index(g)
    known, test, others, hist, t = make_instance(W, g, split, base, args.n_known, args.target_weight)
    tmp = outdir / f"data_{args.tag}_tmp.npz"
    np.savez_compressed(tmp, **{f"hist_{h}": X for h, X in zip(others, hist)},
                        target_all=W[g], known_idx=known, test_idx=test, t=t)
    os.replace(tmp, outdir / "data.npz")

    def seeds_for(fold: int) -> Dict[str, Any]:
        return {"bary": sub_seed(base, PURPOSE["bary"], gi, split, *STREAM_KEY, fold),
                "rkhs": sub_seed(base, PURPOSE["rkhs"], gi, split, *STREAM_KEY, fold) % (2 ** 31 - 1),
                "rkhs_outdir": str(outdir / "rkhs")}

    # leave-one-out cross-validation over the known target weeks
    rows = []
    for fold, vj in enumerate(known):
        refs = hist + [W[g][[k for k in known if k != vj]]]
        for r in solve_all(refs, t, cp_, args, seeds_for(fold)):
            r.update(fold=fold, val=cost(W[g][[vj]], r["w"], cp_) if (r["w"] is not None and is_ok(r["status"])) else np.nan)
            rows.append(r)
    G = pd.DataFrame(rows)
    info = dict(target=g, split=split, seed=base, n_known=args.n_known)
    for k, v in info.items():
        G[k] = v
    G["termination"] = G["aux"].apply(lambda v: v.get("rkhs_termination") if isinstance(v, dict) else None)
    G[["target", "split", "method", "h", "fold", "status", "termination", "objective", "solver", "val",
       "solve_time"]].to_csv(outdir / f"cv_grid_{args.tag}.csv", index=False)
    cv = (G.groupby(["method", "h"]).agg(cv_cost=("val", "mean"), n_ok=("val", lambda s: int(s.notna().sum())))
          .reset_index())
    cv = cv[cv.n_ok == len(known)]          # a hyperparameter must solve in every fold

    def pick(grp: pd.DataFrame) -> pd.DataFrame:
        # validation costs within a relative tolerance of the best count as ties (solver
        # round-off); among ties the smallest hyperparameter is selected
        best = grp.cv_cost.min()
        return grp[grp.cv_cost <= best + args.cv_tie_tol * max(1.0, abs(best))].sort_values("h").head(1)

    picks = [pick(grp) for _, grp in cv.groupby("method", sort=True)]
    sel = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame(columns=["method", "h", "cv_cost", "n_ok"])

    # final fit with all known weeks at the selected hyperparameters
    refs = hist + [W[g][known]]
    final = pd.DataFrame(solve_all(refs, t, cp_, args, seeds_for(len(known)), only=dict(zip(sel.method, sel.h))),
                         columns=["method", "h", "status", "objective", "w", "solve_time", "solver", "aux", "rho"])
    final["test_cost"] = [cost(W[g][test], w, cp_) if (w is not None and is_ok(s)) else np.nan
                          for w, s in zip(final["w"], final["status"])]
    sel = sel.merge(G.groupby(["method", "h"]).solve_time.mean().rename("cv_mean_solve_time").reset_index(),
                    on=["method", "h"], how="left")
    res = sel.merge(final[["method", "h", "status", "objective", "w", "test_cost", "solve_time", "solver", "aux", "rho"]],
                    on=["method", "h"], how="left")
    # a method none of whose hyperparameters solved in every fold is recorded without a test cost
    missing = [m for m in args.methods if m not in set(res.method)]
    res = pd.concat([res, pd.DataFrame([dict(method=m, status="failed: no hyperparameter solved in every fold")
                                        for m in missing])], ignore_index=True)
    for k, v in info.items():
        res[k] = v
    for c in ("w", "aux", "rho"):
        res[c] = res[c].apply(lambda v: json.dumps(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "")
    res.to_csv(outdir / f"selected_{args.tag}_tmp.csv", index=False)
    os.replace(outdir / f"selected_{args.tag}_tmp.csv", done_flag)
    return f"done {outdir.name}"


def summarize(root: Path) -> pd.DataFrame:
    files = sorted(f for f in root.glob("instances/*/selected_*.csv") if not f.name.endswith("_tmp.csv"))
    if not files:
        return pd.DataFrame()
    S = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    S = S.drop_duplicates(subset=["target", "split", "method"], keep="last")
    S.to_csv(root / "all_selected.csv", index=False)
    wide = S.pivot_table(index="method", columns="target", values="test_cost", aggfunc="mean")
    wide = wide.reindex([m for m in METHODS if m in wide.index])
    wide.round(2).to_csv(root / "summary_mean_test_cost.csv")
    return wide


def environment_info() -> Dict[str, Any]:
    """Versions and hardware, stored with every run."""
    import platform
    info: Dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform(),
                            "cpu_count": os.cpu_count()}
    for mod in ("numpy", "scipy", "pandas", "cvxpy", "mosek", "gurobipy", "sklearn"):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", None)
            if v is None and mod == "mosek":
                v = ".".join(str(x) for x in m.Env.getversion())
            if v is None and mod == "gurobipy":
                v = ".".join(str(x) for x in m.gurobi.version())
            info[mod] = str(v)
        except Exception as exc:  # noqa: BLE001
            info[mod] = f"unavailable ({type(exc).__name__})"
    return info


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_xlsx", default=str(RAW_COFFEE_XLSX), help="Maven Roasters transaction workbook")
    p.add_argument("--outdir", default="", help="result folder (default: results/run_<timestamp>)")
    p.add_argument("--methods", default=",".join(CONIC_METHODS),
                   help=f"comma-separated subset of {','.join(METHODS)}; jobs with different methods can share --outdir")
    p.add_argument("--targets", default="all", help="target products, e.g. group_1,group_3 (default: all five)")
    p.add_argument("--n_splits", type=int, default=10, help="number of repetitions")
    p.add_argument("--split_ids", default="", help="run only these repetitions, e.g. 0-4 (default: all)")
    p.add_argument("--n_known", type=int, default=5, help="known target weeks; the remaining weeks are test weeks")
    p.add_argument("--target_weight", type=float, default=0.5, help="weight of the target reference")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--delta_grid", default=DELTA_GRID, help="grid of delta (affine, static)")
    p.add_argument("--epsilon_grid", default=DELTA_GRID, help="grid of the radius (pool, mixture, minkowski, barycenter)")
    p.add_argument("--nu_grid", default=NU_GRID, help="grid of nu (rkhs)")
    p.add_argument("--cv_tie_tol", type=float, default=1e-6, help="relative tolerance for ties in the validation cost")
    p.add_argument("--barycenter_num_samples", type=int, default=128, help="samples from the Gelbrich barycenter")
    p.add_argument("--rkhs_n_J", type=int, default=2000)
    p.add_argument("--rkhs_T", type=int, default=50)
    p.add_argument("--rkhs_time_limit", type=int, default=3600)
    p.add_argument("--cvxpy_solver", default="MOSEK")
    p.add_argument("--solver_threads", type=int, default=1, help="MOSEK threads per worker (0 = MOSEK default)")
    p.add_argument("--gurobi_threads", type=int, default=1, help="Gurobi threads per RKHS solve (0 = Gurobi default)")
    p.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    p.add_argument("--mosek_license_file", default="")
    p.add_argument("--gurobi_license_file", default="")
    p.add_argument("--summarize_only", action="store_true", help="only rebuild the summary of --outdir")
    return p


def main() -> None:
    a = build_parser().parse_args()
    methods = parse_list(a.methods)
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise ValueError(f"unknown methods {bad}; choose from {METHODS}")
    tag = "rkhs" if methods == ["rkhs"] else ("conic" if sorted(methods) == sorted(CONIC_METHODS) else "_".join(sorted(methods)))
    args = dict(methods=methods, tag=tag, seed=a.seed, n_known=a.n_known, target_weight=a.target_weight,
                delta_grid=parse_float_list(a.delta_grid), epsilon_grid=parse_float_list(a.epsilon_grid),
                nu_grid=parse_float_list(a.nu_grid), cv_tie_tol=a.cv_tie_tol,
                barycenter_num_samples=a.barycenter_num_samples, rkhs_n_J=a.rkhs_n_J, rkhs_T=a.rkhs_T,
                rkhs_time_limit=a.rkhs_time_limit, cvxpy_solver=a.cvxpy_solver, solver_threads=a.solver_threads,
                gurobi_threads=a.gurobi_threads, gurobi_license_file=a.gurobi_license_file)
    root = Path(a.outdir) if a.outdir else Path(__file__).resolve().parent / "results" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    root.mkdir(parents=True, exist_ok=True)
    if a.summarize_only:
        print(summarize(root).to_string())
        return
    with open(root / f"config_{tag}_{datetime.now():%Y%m%d_%H%M%S}.json", "w") as f:
        json.dump(dict(vars(a), resolved=args, environment=environment_info()), f, indent=2)

    W, prices = load_weekly_demand(a.raw_xlsx)
    cp_ = cost_parameters(prices)
    print(f"products {sorted(W)} | weeks {[len(v) for v in W.values()]}", flush=True)
    print(f"costs: p={prices.tolist()} c={cp_['C'].tolist()} h={np.round(cp_['H'], 4).tolist()} "
          f"b={np.round(cp_['B'], 4).tolist()}", flush=True)
    targets = sorted(W) if a.targets == "all" else parse_list(a.targets)
    if a.split_ids.strip() and "-" in a.split_ids and not a.split_ids.strip().startswith("["):
        lo, hi = (int(v) for v in a.split_ids.split("-"))
        split_ids = list(range(lo, hi + 1))
    elif a.split_ids.strip():
        split_ids = [int(v) for v in parse_float_list(a.split_ids)]
    else:
        split_ids = list(range(a.n_splits))
    W_list = {g: v.tolist() for g, v in W.items()}
    jobs = [dict(outdir=str(root / "instances" / f"{g}_split{sp}"), W=W_list, target=g, split=sp, cost_params=cp_)
            for g in targets for sp in split_ids]

    if any(m != "rkhs" for m in methods):          # fail fast if the conic solver is not usable
        import cvxpy as cp
        x = cp.Variable(2)
        cp.Problem(cp.Minimize(cp.norm(x - 1)), [x >= 0]).solve(solver=a.cvxpy_solver)
    print(f"{len(jobs)} instances -> {root}", flush=True)
    t0 = time.time()
    if a.workers <= 1:
        for j in jobs:
            try:
                print(run_instance(j, args), f"[{time.time() - t0:.0f}s]", flush=True)
            except Exception:  # noqa: BLE001
                print("FAILED", j["outdir"], traceback.format_exc(), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(run_instance, j, args): j for j in jobs}
            for f in as_completed(futs):
                try:
                    print(f.result(), f"[{time.time() - t0:.0f}s]", flush=True)
                except Exception:  # noqa: BLE001
                    print("FAILED", futs[f]["outdir"], traceback.format_exc(), flush=True)
    wide = summarize(root)
    print(wide.to_string() if len(wide) else "no results", flush=True)


if __name__ == "__main__":
    main()
