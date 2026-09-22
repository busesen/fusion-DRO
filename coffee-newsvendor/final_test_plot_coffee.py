from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Dict, List

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from coffee_common import parse_list

VAL_COL = "test_expected_cost (validation 10%)"
TEST_COL = "test_expected_cost (unseen 90%)"
PCT_COL = "pct_improvement_vs_ERM (unseen 90%)"

METHOD_STYLE = {
    "affine": {"color": "#1f77b4", "label": "affine", "marker": "o"},
    "static": {"color": "#17becf", "label": "static", "marker": "o"},
    "pooled": {"color": "#9467bd", "label": "pooled", "marker": "v"},
    "mixture": {"color": "#2ca02c", "label": "mixture", "marker": "s"},
    "minkowski": {"color": "#d62728", "label": "minkowski", "marker": "D"},
    "barycenter_fitted": {"color": "#ff7f0e", "label": "barycenter fitted", "marker": "^"},
    "rkhs": {"color": "#6b8e23", "label": "rkhs", "marker": "P"},
}
METHOD_ORDER = ["affine", "static", "pooled", "mixture", "minkowski", "barycenter_fitted", "rkhs", "erm"]

HEADER_FILL = PatternFill(fill_type="solid", fgColor="D9EAF7")
BEST_FILL = PatternFill(fill_type="solid", fgColor="FFF2CC")
BLOCK_FILL = PatternFill(fill_type="solid", fgColor="E2F0D9")
HEADER_FONT = Font(bold=True)
BEST_FONT = Font(bold=True)
BLOCK_FONT = Font(bold=True)
CENTER_ALIGN = Alignment(horizontal="center", vertical="center")
SEPARATOR_SIDE = Side(style="medium", color="808080")


def _results_timestamp(path: str | Path) -> str | None:
    for candidate in [Path(path), *Path(path).parents]:
        match = re.search(r"(\d{8}_\d{6})(?:_\d+)?$", candidate.name)
        if match:
            return match.group(1)
    return None


def _selection_summary_path(outdir: str | Path) -> Path:
    outdir = Path(outdir)
    timestamp = _results_timestamp(outdir)
    if timestamp is None:
        return outdir / "selection_summary.xlsx"
    return outdir / f"selection_summary_{timestamp}.xlsx"


def _cleanup_selection_summaries(outdir: str | Path, keep_path: str | Path) -> None:
    outdir = Path(outdir)
    keep_path = Path(keep_path).resolve()
    for candidate in outdir.glob("selection_summary*.xlsx"):
        if candidate.resolve() == keep_path:
            continue
        if candidate.is_file():
            candidate.unlink()


def _pretty_sources(raw: str) -> str:
    vals = parse_list(raw)
    return ", ".join(vals) if vals else "(none)"


def _radius_text(row: pd.Series) -> str:
    method = str(row.get("method", ""))
    if method in {"affine", "static"} and pd.notna(row.get("delta")):
        return f"delta={float(row['delta']):.6g}"
    if method == "minkowski":
        if str(row.get("row_kind", "")) == "aff" and pd.notna(row.get("delta")):
            return f"delta={float(row['delta']):.6g}"
        if pd.notna(row.get("epsilon_single")):
            return f"epsilon={float(row['epsilon_single']):.6g}"
    if method in {"pooled", "mixture", "barycenter_fitted"} and pd.notna(row.get("epsilon_single")):
        return f"epsilon={float(row['epsilon_single']):.6g}"
    if method == "rkhs" and pd.notna(row.get("nu")):
        return f"nu={float(row['nu']):.6g}"
    return "-"


def _selected_value_text(block: pd.DataFrame) -> str:
    method = str(block["method"].iloc[0]) if not block.empty else ""
    if method in {"affine", "static"}:
        vals = [f"{float(v):.6g}" for v in pd.to_numeric(block["delta"], errors="coerce").dropna().tolist()]
        return ", ".join(vals) if vals else "-"
    if method in {"pooled", "mixture", "minkowski", "barycenter_fitted"}:
        vals = [f"{float(v):.6g}" for v in pd.to_numeric(block["epsilon_single"], errors="coerce").dropna().tolist()]
        return ", ".join(vals) if vals else "-"
    if method == "rkhs":
        vals = [f"{float(v):.6g}" for v in pd.to_numeric(block["nu"], errors="coerce").dropna().tolist()]
        return ", ".join(vals) if vals else "-"
    return "-"


def _selected_radius_text(block: pd.DataFrame) -> str:
    return _selected_value_text(block)


def _select_best_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "run_seed" not in out.columns:
        out["run_seed"] = 1234
    if "row_kind" in out.columns:
        out = out[~(out["method"].astype(str).eq("minkowski") & out["row_kind"].astype(str).eq("aff"))].copy()
    group_cols = ["target_source", "selected_historical", "source_values", "K_total", "method", "run_seed"]
    out = (
        out.sort_values(group_cols + [VAL_COL, "delta", "epsilon_single", "nu"], kind="stable")
        .groupby(group_cols, as_index=False, sort=False)
        .first()
    )
    out["_method_rank"] = out["method"].astype(str).map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(999)
    return out.sort_values(["target_source", "K_total", "selected_historical", "_method_rank", "run_seed"], kind="stable").drop(columns=["_method_rank"]).reset_index(drop=True)


def _attach_tuning_time_columns(selected: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    out = selected.copy()
    if "empirical_success" not in out.columns:
        out["empirical_success"] = False
    if "empirical_runtime_sec" not in out.columns:
        out["empirical_runtime_sec"] = np.nan
    if "additional_solve_time_sec" not in out.columns:
        out["additional_solve_time_sec"] = np.nan
    if "shared_precompute_sec" not in out.columns:
        out["shared_precompute_sec"] = np.nan

    success = out["empirical_success"].fillna(False).astype(bool)
    selected_runtime = pd.to_numeric(out["empirical_runtime_sec"], errors="coerce")
    additional_runtime = (
        pd.to_numeric(out["additional_solve_time_sec"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["shared_precompute_sec"], errors="coerce").fillna(0.0)
    )

    out["tuning_runtime_sec"] = np.where(success, selected_runtime + additional_runtime, np.nan)
    out["additional_solve_time_included_sec"] = np.where(success, additional_runtime, np.nan)
    return out


def _attach_pct_vs_erm(selected: pd.DataFrame) -> pd.DataFrame:
    out = selected.copy()
    key_cols = ["target_source", "selected_historical", "source_values", "K_total", "run_seed"]
    erm = out[out["method"].astype(str) == "erm"][key_cols + [TEST_COL]].rename(columns={TEST_COL: "erm_test_cost"})
    out = out.merge(erm, on=key_cols, how="left")
    out[PCT_COL] = 100.0 * (pd.to_numeric(out["erm_test_cost"], errors="coerce") - pd.to_numeric(out[TEST_COL], errors="coerce")) / pd.to_numeric(out["erm_test_cost"], errors="coerce")
    out.loc[out["method"].astype(str) == "erm", PCT_COL] = 0.0
    return out


def _safe_std(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return 0.0
    return float(np.std(vals, ddof=1))


def _safe_mean(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def _summarize(selected: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = ["target_source", "selected_historical", "source_values", "K_total", "method"]
    for _, block in selected.groupby(group_cols, sort=False, dropna=False):
        block = block.sort_values("run_seed", kind="stable")
        n = int(block["run_seed"].nunique()) if "run_seed" in block.columns else int(block.shape[0])
        mean_oos = _safe_mean(block[TEST_COL])
        std_oos = _safe_std(block[TEST_COL])
        mean_val = _safe_mean(block[VAL_COL])
        std_val = _safe_std(block[VAL_COL])
        time_col = "tuning_runtime_sec" if "tuning_runtime_sec" in block.columns else "empirical_runtime_sec"
        additional_col = "additional_solve_time_included_sec" if "additional_solve_time_included_sec" in block.columns else None
        mean_time = _safe_mean(block[time_col])
        std_time = _safe_std(block[time_col])
        mean_additional_time = _safe_mean(block[additional_col]) if additional_col is not None else float("nan")
        std_additional_time = _safe_std(block[additional_col]) if additional_col is not None else float("nan")
        mean_pct = _safe_mean(block[PCT_COL])
        std_pct = _safe_std(block[PCT_COL])
        rows.append(
            {
                "target": str(block["target_source"].iloc[0]),
                "sources": _pretty_sources(str(block["selected_historical"].iloc[0])),
                "K_total": int(block["K_total"].iloc[0]),
                "method": str(block["method"].iloc[0]),
                "n_seeds": n,
                "selected_radius": _selected_radius_text(block),
                "mean oos": mean_oos,
                "std of oos": std_oos,
                "ci95 of oos": float(1.96 * std_oos / np.sqrt(n)) if n > 0 else float("nan"),
                "mean validation": mean_val,
                "std validation": std_val,
                "mean solve time (sec)": mean_time,
                "std solve time (sec)": std_time,
                "additional solve time (included)": mean_additional_time,
                "std additional solve time (sec)": std_additional_time,
                "mean pct improvement vs ERM": mean_pct,
                "std pct improvement vs ERM": std_pct,
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["_method_rank"] = summary["method"].astype(str).map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(999)
    return summary.sort_values(["target", "sources", "K_total", "_method_rank"], kind="stable").drop(columns=["_method_rank"]).reset_index(drop=True)


def _best_mask(values: pd.Series, prefer: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    out = pd.Series(False, index=values.index)
    if not valid.any():
        return out
    best = numeric[valid].max() if prefer == "max" else numeric[valid].min()
    out[valid] = np.isclose(numeric[valid], best, rtol=1e-9, atol=1e-12)
    return out


def _style_header_row(ws) -> None:
    for cell in ws[1]:
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = CENTER_ALIGN


def _set_column_widths(ws) -> None:
    widths: Dict[int, int] = {}
    for row in ws.iter_rows(values_only=True):
        for idx, value in enumerate(row, start=1):
            width = len(str(value)) if value is not None else 0
            widths[idx] = max(widths.get(idx, 0), width)
    for idx, width in widths.items():
        ws.column_dimensions[ws.cell(row=1, column=idx).column_letter].width = min(max(width + 2, 10), 40)


def _highlight_oos_table(workbook_path: Path, sheet_name: str = "oos") -> None:
    wb = openpyxl.load_workbook(workbook_path)
    ws = wb[sheet_name]
    _style_header_row(ws)
    header = {ws.cell(row=1, column=c).value: c for c in range(1, ws.max_column + 1)}

    # Build contiguous row blocks rather than grouping by key globally. This
    # keeps merges well-formed even when a sheet was reloaded from previously
    # merged cells or resorted.
    blocks: List[List[int]] = []
    current_rows: List[int] = []
    current_key = None
    for r in range(2, ws.max_row + 1):
        key = (
            ws.cell(row=r, column=header["target"]).value,
            ws.cell(row=r, column=header["sources"]).value,
            ws.cell(row=r, column=header["K_total"]).value,
        )
        if current_key is None or key == current_key:
            current_rows.append(r)
            current_key = key
        else:
            if current_rows:
                blocks.append(current_rows)
            current_rows = [r]
            current_key = key
    if current_rows:
        blocks.append(current_rows)

    metric_prefs = {
        "mean oos": "min",
        "std of oos": "min",
        "mean solve time (sec)": "min",
        "additional solve time (included)": "min",
        "std solve time (sec)": "min",
    }

    for row_ids in blocks:
        start_row = min(row_ids)
        end_row = max(row_ids)
        for col_name, prefer in metric_prefs.items():
            if col_name not in header:
                continue
            values = pd.Series([ws.cell(row=r, column=header[col_name]).value for r in row_ids], index=row_ids)
            mask = _best_mask(values, prefer=prefer)
            for r, is_best in mask.items():
                if bool(is_best):
                    cell = ws.cell(row=int(r), column=header[col_name])
                    cell.fill = BEST_FILL
                    cell.font = BEST_FONT
        for col_name in ["target", "sources", "K_total"]:
            if col_name in header:
                for r in row_ids:
                    cell = ws.cell(row=r, column=header[col_name])
                    cell.fill = BLOCK_FILL
                    cell.font = BLOCK_FONT
                    cell.alignment = CENTER_ALIGN
                if len(row_ids) > 1:
                    col_idx = header[col_name]
                    ws.merge_cells(
                        start_row=start_row,
                        end_row=end_row,
                        start_column=col_idx,
                        end_column=col_idx,
                    )
                    merged_cell = ws.cell(row=start_row, column=col_idx)
                    merged_cell.fill = BLOCK_FILL
                    merged_cell.font = BLOCK_FONT
                    merged_cell.alignment = CENTER_ALIGN
        for c in range(1, ws.max_column + 1):
            ws.cell(row=end_row, column=c).border = Border(bottom=SEPARATOR_SIDE)

    ws.freeze_panes = "A2"
    _set_column_widths(ws)
    wb.save(workbook_path)


def _plot(summary: pd.DataFrame, out_pdf: Path) -> None:
    if summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(f"Plot skipped: matplotlib is not installed; summary workbook was still written to {out_pdf.parent}")
        return
    plot_rows = (
        summary.sort_values(["K_total", "method", "mean validation"], kind="stable")
        .groupby(["K_total", "method"], as_index=False, sort=False)
        .first()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHOD_ORDER:
        if method == "erm":
            continue
        sub = plot_rows[plot_rows["method"].astype(str) == method].sort_values("K_total")
        if sub.empty:
            continue
        style = METHOD_STYLE.get(method, {"color": "#444444", "label": method, "marker": "o"})
        ax.plot(
            pd.to_numeric(sub["K_total"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(sub["mean pct improvement vs ERM"], errors="coerce").to_numpy(dtype=float),
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
            markersize=6,
            label=style["label"],
        )
    ax.axhline(0.0, color="#999999", linewidth=1, linestyle="--")
    ax.set_xlabel("K_total")
    ax.set_ylabel("Relative improvement vs ERM (%)")
    ax.set_title("Coffee Shop: Out-of-sample Improvement vs ERM")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def _display_summary(summary: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "target",
        "sources",
        "K_total",
        "method",
        "n_seeds",
        "selected_radius",
        "mean oos",
        "std of oos",
        "mean solve time (sec)",
        "additional solve time (included)",
        "std solve time (sec)",
    ]
    out = summary.copy()
    missing = [c for c in keep_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing expected summary columns: {missing}")
    return out[keep_cols].copy()


def main() -> None:
    p = argparse.ArgumentParser(description="Select best coffee-shop hyperparameters per seed, summarize OOS performance, and plot unseen-test improvement vs ERM.")
    p.add_argument("--eval_csv", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--k_total_grid", type=str, default="all")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.eval_csv)
    if str(args.k_total_grid).strip().lower() != "all":
        keep = [int(v) for v in str(args.k_total_grid).strip().strip("[]").split(",") if v.strip()]
        df = df[df["K_total"].astype(int).isin(keep)].copy()

    selected = _select_best_per_seed(df)
    selected = _attach_tuning_time_columns(selected, df)
    selected = _attach_pct_vs_erm(selected)
    summary = _summarize(selected)
    summary_display = _display_summary(summary)

    xlsx_path = _selection_summary_path(outdir)
    pdf_path = outdir / "test_pct_improvement_vs_ERM_exact.pdf"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_runs", index=False)
        selected.to_excel(writer, sheet_name="selected_per_seed", index=False)
        summary_display.to_excel(writer, sheet_name="oos", index=False)

    _highlight_oos_table(xlsx_path, "oos")
    _cleanup_selection_summaries(outdir, xlsx_path)
    _plot(summary, pdf_path)

    print(f"Selection summary: {xlsx_path}")
    print(f"Plot: {pdf_path}")
    print(f"Saved under {outdir}")


if __name__ == "__main__":
    main()
