"""Table of Section 5.2 from a run_newsvendor.py result folder.

For every method, the table reports the average daily cost on the test weeks (Test cost) and
the time in seconds of the final solve at the selected hyperparameter (Time), for each target
product, averaged over the repetitions. The lowest value of each metric in each column is in
bold. A csv file with the corresponding standard errors is written next to the table.

    python coffee-newsvendor/make_table.py --results coffee-newsvendor/results/<folder>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from coffee_data import PRODUCT_NAMES

ORDER = ["affine", "static", "static_sq", "minkowski", "mixture", "barycenter", "pool", "erm", "rkhs"]
METRICS = [("test_cost", "Test cost"), ("solve_time", "Time")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, nargs="+",
                    help="result folder(s); for an instance present in several folders, the last one is used")
    ap.add_argument("--out", default="", help="output .tex file (default: <last folder>/table.tex)")
    a = ap.parse_args()

    files = [f for r in a.results for f in sorted(Path(r).glob("instances/*/selected_*.csv")) if not f.name.endswith("_tmp.csv")]
    if not files:
        raise SystemExit("no results found")
    S = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    S = S.drop_duplicates(["target", "split", "method"], keep="last")
    targets = sorted(S.target.unique())
    methods = [m for m in ORDER if m in set(S.method)]
    reps = int(S.groupby("target").split.nunique().min())

    def mean(m, t, col):
        v = S[(S.method == m) & (S.target == t)][col]
        return float(v.mean()) if v.notna().any() else np.nan

    val = {(col, m): [mean(m, t, col) for t in targets] for col, _ in METRICS for m in methods}
    best = {col: [np.nanmin([val[(col, m)][j] for m in methods]) for j in range(len(targets))] for col, _ in METRICS}

    def fmt(v, col, j):
        if not np.isfinite(v):
            return "--"
        s = f"{v:.2f}"
        return rf"\textbf{{{s}}}" if s == f"{best[col][j]:.2f}" else s

    lines = [r"\begin{table}[t]", r"\centering", r"\fontsize{8}{9.5}\selectfont", r"\setlength{\tabcolsep}{3.5pt}",
             r"\renewcommand{\arraystretch}{0.95}", r"\begin{tabular}{@{}ll" + "r" * len(targets) + "@{}}", r"\toprule",
             "Method & Metric & " + " & ".join(PRODUCT_NAMES.get(t, t) for t in targets) + r" \\", r"\midrule"]
    for i, m in enumerate(methods):
        for k, (col, label) in enumerate(METRICS):
            head = rf"\multirow{{{len(METRICS)}}}{{*}}{{\textsc{{{m}}}}}" if k == 0 else ""
            end = r" \\" + (r" \midrule" if k == len(METRICS) - 1 and i < len(methods) - 1 else "")
            lines.append(f"{head} & {label} & " + " & ".join(fmt(v, col, j) for j, v in enumerate(val[(col, m)])) + end)
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Average daily cost on the test observations and time in seconds to solve the final problem "
              r"with the selected hyperparameter, excluding cross-validation, for each method and target product, "
              rf"averaged over ${reps}$ repetitions. Bold entries mark the best value per column.}}",
              r"\label{tab:coffee}", r"\end{table}"]
    tex = "\n".join(lines)
    out = Path(a.out) if a.out else Path(a.results[-1]) / "table.tex"
    out.write_text(tex)

    rows = []
    for m in methods:
        for t in targets:
            sub = S[(S.method == m) & (S.target == t)]
            r = dict(method=m, target=PRODUCT_NAMES.get(t, t), n=len(sub))
            for col, _ in METRICS:
                r[f"{col}_mean"] = sub[col].mean()
                r[f"{col}_se"] = sub[col].std() / np.sqrt(max(sub[col].notna().sum(), 1))
            rows.append(r)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    bad = S[~S.status.astype(str).str.startswith("optimal")]
    if len(bad):
        print(f"WARNING: {len(bad)} selected models did not solve to optimality:")
        print(bad[["method", "target", "split", "h", "status"]].to_string(index=False))
    print(tex)
    print(f"\nrepetitions per target: {reps}\nwritten: {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
