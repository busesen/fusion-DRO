"""Table 3 of Section 5.3: mean suboptimality 100 (CE / CE_ORACLE - 1) over the simulations, with standard errors.

    python credit-classification/make_table.py --results credit-classification/results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["affine", "static", "minkowski", "moment", "tv"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    a = ap.parse_args()
    S = pd.concat([pd.read_csv(f) for f in sorted(a.results.glob("n*/*_sims*.csv"))], ignore_index=True)
    oracle = S[S.method == "oracle"].set_index(["n", "sim"]).test_ce
    S = S[S.method != "oracle"].join(oracle.rename("oracle_ce"), on=["n", "sim"]).dropna(subset=["oracle_ce"])
    S["subopt"] = 100.0 * (S.test_ce / S.oracle_ce - 1.0)
    T = S.groupby(["n", "method"]).subopt.agg(mean="mean", se=lambda v: v.std(ddof=1) / np.sqrt(len(v)), sims="count")
    T.reset_index().to_csv(a.results / "table.csv", index=False)
    lines = [r"\begin{tabular}{r*{5}{r@{\,}l}}", r"\toprule",
             "$N$ & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{\textsc{{{c}}}}}" for c in COLS) + r" \\", r"\midrule"]
    for n in sorted(S.n.unique()):
        vals = {c: T.loc[(n, c)] for c in COLS if (n, c) in T.index}
        best = min(round(v["mean"], 2) for v in vals.values())
        fmt = lambda v: (rf"\textbf{{{v['mean']:.2f}}}" if round(v["mean"], 2) == best else f"{v['mean']:.2f}") \
            + f" & ({v['se']:.2f})"
        cells = [fmt(vals[c]) if c in vals else "-- &" for c in COLS]
        lines.append(f"${n:,}$".replace(",", "{,}") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (a.results / "table.tex").write_text("\n".join(lines) + "\n")
    print(T.round(2).to_string())


if __name__ == "__main__":
    main()
