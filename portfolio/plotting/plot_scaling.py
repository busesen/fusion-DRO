from __future__ import annotations

from pathlib import Path
import shutil
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_PLOT_STYLE_INITIALIZED = False


def _configure_plot_style() -> None:
    global _PLOT_STYLE_INITIALIZED
    if _PLOT_STYLE_INITIALIZED:
        return

    base_style = {
        "font.size": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
    }
    plt.rcParams.update(base_style)

    if shutil.which("latex") is not None:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
        })
        plt.rcParams["text.latex.preamble"] = r"""\usepackage{amsmath,amsfonts,amssymb} \usepackage{times}"""
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        })
        plt.rcParams["text.latex.preamble"] = ""
        print("[plot_scaling] latex not found; falling back to Matplotlib text rendering.", flush=True)

    _PLOT_STYLE_INITIALIZED = True


METHOD_ORDER = ['affine', 'static', 'mixture', 'minkowski', 'exact']
METHOD_STYLE = {
    'affine': {'color': '#fbb13c', 'linestyle': '-', 'marker': 'h', 'markersize':8, 'linewidth': 2.0},
    'static': {'color': '#73d2de', 'linestyle': '-.', 'marker': 's','markersize':7, 'linewidth': 2.0},
    'mixture': {'color': '#218380', 'linestyle': '--', 'marker': '^','markersize':8, 'linewidth': 2.2},
    'minkowski': {'color': '#d81159', 'linestyle': ':', 'marker': 'D','markersize':6, 'linewidth': 2.4},
    'exact': {'color': '#8f2d56', 'linestyle': '-', 'marker': '*','markersize':10, 'linewidth': 2.0},
}
CI_Z = 1.96


def _method_label(method: str) -> str:
    """Method name in small capitals, as in the paper (plain text without LaTeX)."""
    return rf"\textsc{{{method}}}" if plt.rcParams["text.usetex"] else str(method)

# Joint log-time figure (`time_vs_log_joint.pdf`) styling.
JOINT_LOG_FIGSIZE = (14, 6)
JOINT_LOG_AXIS_LABELSIZE = 22      # x/y axis-label size for the two joint log panels
JOINT_LOG_TICK_LABELSIZE = 20      # x/y tick-label size for the two joint log panels
JOINT_LOG_LEGEND_FONTSIZE = 20     # shared legend size for the joint log figure
JOINT_LOG_LEGEND_BORDERPAD = 0.15   # white space between legend frame and contents
JOINT_LOG_LEGEND_HANDLETEXTPAD = 0.3  # gap between a line/marker and its label text
JOINT_LOG_LEGEND_COLUMNSPACING = 0.6  # horizontal gap between legend columns
JOINT_LOG_LEGEND_LABELSPACING = 0.5   # vertical gap between legend rows
JOINT_LOG_GRID_ALPHA = 0.25        # grid opacity in the joint log figure
JOINT_LOG_W_PAD = 0.3              # horizontal white space between the two joint log panels
JOINT_LOG_LAYOUT_RECT = (0.0, 0.0, 1.0, 0.92)  # leaves room above the panels for the shared legend

# Joint objective figure (`objective_vs_joint.pdf`) styling.
JOINT_OBJECTIVE_FIGSIZE = (14, 6)
JOINT_OBJECTIVE_AXIS_LABELSIZE = 22
JOINT_OBJECTIVE_TICK_LABELSIZE = 20
JOINT_OBJECTIVE_LEGEND_FONTSIZE = 20
JOINT_OBJECTIVE_LEGEND_BORDERPAD = 0.13
JOINT_OBJECTIVE_LEGEND_HANDLETEXTPAD = 0.3
JOINT_OBJECTIVE_LEGEND_COLUMNSPACING = 0.6
JOINT_OBJECTIVE_LEGEND_LABELSPACING = 0.5
JOINT_OBJECTIVE_GRID_ALPHA = 0.25
JOINT_OBJECTIVE_W_PAD = 0.3
JOINT_OBJECTIVE_LAYOUT_RECT = (0.0, 0.0, 1.0, 0.92)

# Optional manual y-limits for the main scaling plots.
# Set any entry to a tuple `(ymin, ymax)` and rerun `replot`.
# Use `None` for either endpoint to keep the automatically chosen bound.
# Leave the whole entry as `None` to use Matplotlib's automatic limits.
MANUAL_YLIMS: dict[str, tuple[float | None, float | None] | None] = {
    # Joint log-time figure: left = vs K, right = vs N_k.
    'time_vs_log_joint_left': None,
    'time_vs_log_joint_right': None,
    # Joint objective figure: left = vs K, right = vs N_k.
    'objective_vs_joint_left': (0.25,0.6),
    # 'objective_vs_joint_left': None,
    'objective_vs_joint_right': (0.15,0.45),
    # 'objective_vs_joint_right': None,
}

# Optional manual x-limits for the scaling plots.
# Set any entry to `(xmin, xmax)` and rerun `replot`.
# Use `None` for either endpoint to keep the automatically chosen bound.
MANUAL_XLIMS: dict[str, tuple[float | None, float | None] | None] = {
    # Joint log-time figure: left = vs K, right = vs N_k.
    'time_vs_log_joint_left': None,
    'time_vs_log_joint_right': None,
    # Joint objective figure: left = vs K, right = vs N_k.
    'objective_vs_joint_left': None,
    'objective_vs_joint_right': None,
}
# Legend positions use figure coordinates, not panel coordinates.
MANUAL_LEGEND_KWARGS: dict[str, dict[str, object] | None] = {
    'time_vs_log_joint': {'loc': 'upper center', 'bbox_to_anchor': (0.5, 0.97)},
    'objective_vs_joint': {'loc': 'upper center', 'bbox_to_anchor': (0.5, 0.97)},
}


def _aggregate_mean_ci(clean: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    grouped = (
        clean.groupby(['method', x_col])[y_col]
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .sort_values([x_col, 'method'])
    )
    if grouped.empty:
        return grouped
    grouped['std'] = grouped['std'].fillna(0.0)
    grouped['count'] = grouped['count'].clip(lower=1)
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['ci_half'] = CI_Z * grouped['se']
    grouped['ci_low'] = grouped['mean'] - grouped['ci_half']
    grouped['ci_high'] = grouped['mean'] + grouped['ci_half']
    return grouped


def _prepare_grouped_curve_data(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    sweep_axis: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]] | None:
    if x_col not in df.columns or y_col not in df.columns:
        return None
    clean = df.copy()
    if sweep_axis is not None and 'sweep_axis' in clean.columns:
        clean = clean[clean['sweep_axis'] == sweep_axis]
    if y_col == 'solve_time_sec' and 'status' in clean.columns:
        status_str = clean['status'].fillna('').astype(str)
        bad = status_str.str.startswith('skipped_') | status_str.str.startswith('no_usable_solution')
        clean.loc[bad, y_col] = float('nan')
    clean = clean.dropna(subset=[x_col, y_col])
    if clean.empty:
        return None
    grouped = _aggregate_mean_ci(clean, x_col=x_col, y_col=y_col)
    if grouped.empty:
        return None
    x_unique = np.sort(grouped[x_col].dropna().astype(float).unique())
    methods = list(grouped['method'].unique())
    methods = sorted(methods, key=lambda method: METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER))
    return grouped, x_unique, methods


def _set_discrete_x_ticks(ax, x_values: np.ndarray, integer_stride: int | None = None) -> None:
    if x_values.size == 0:
        return
    x_unique = np.sort(np.unique(x_values.astype(float)))
    is_integral = np.allclose(x_unique, np.round(x_unique), atol=1e-9)
    if is_integral and x_unique.size <= 25:
        tick_vals = x_unique
        if integer_stride is not None and integer_stride > 1:
            start = int(round(x_unique.min()))
            tick_vals = np.array(
                [v for v in x_unique if (int(round(v)) - start) % int(integer_stride) == 0],
                dtype=float,
            )
            if tick_vals.size == 0:
                tick_vals = x_unique
        ax.set_xticks(tick_vals)
        ax.set_xticklabels([str(int(round(v))) for v in tick_vals])
    else:
        ax.set_xticks(x_unique)
    tick_labelsize: int | None = None
    if x_unique.size >= 15:
        tick_labelsize = 18
    elif x_unique.size >= 10:
        tick_labelsize = 20
    if tick_labelsize is not None:
        ax.tick_params(axis='both', labelsize=tick_labelsize)
    ax.minorticks_off()


def _apply_main_ylim(ax, plot_key: str) -> None:
    manual = MANUAL_YLIMS.get(plot_key)
    current_low, current_high = ax.get_ylim()
    is_log = ax.get_yscale() == 'log'
    force_lower_tick = False
    force_upper_tick = False
    if manual is not None:
        lower, upper = manual
        y_low = current_low if lower is None else float(lower)
        y_high = current_high if upper is None else float(upper)
        ax.set_ylim(y_low, y_high)
        force_lower_tick = (lower is not None) and (not is_log)
        force_upper_tick = (upper is not None) and (not is_log)

    y_low, y_high = ax.get_ylim()
    if not (np.isfinite(y_low) and np.isfinite(y_high)):
        return

    ticks = np.asarray(ax.get_yticks(), dtype=float)
    ticks = ticks[np.isfinite(ticks)]
    tol = 1e-9 * max(1.0, abs(y_high - y_low))
    ticks = ticks[(ticks >= y_low - tol) & (ticks <= y_high + tol)]
    if str(plot_key).startswith('time_vs_') and (not is_log) and y_low < 0.0 < y_high:
        if not np.any(np.isclose(ticks, 0.0, atol=tol, rtol=0.0)):
            ticks = np.append(ticks, 0.0)
    if force_lower_tick and not np.any(np.isclose(ticks, y_low, atol=tol, rtol=0.0)):
        ticks = np.append(ticks, y_low)
    if force_upper_tick and not np.any(np.isclose(ticks, y_high, atol=tol, rtol=0.0)):
        ticks = np.append(ticks, y_high)
    ticks = np.unique(np.round(ticks, 12))
    ticks.sort()
    ax.set_yticks(ticks)
    ax.set_ylim(y_low, y_high)


def _apply_main_xlim(ax, plot_key: str) -> None:
    manual = MANUAL_XLIMS.get(plot_key)
    if manual is None:
        return
    current_low, current_high = ax.get_xlim()
    lower, upper = manual
    x_low = current_low if lower is None else float(lower)
    x_high = current_high if upper is None else float(upper)
    ax.set_xlim(x_low, x_high)


def _plot_grouped_lines(
    ax,
    *,
    grouped: pd.DataFrame,
    x_col: str,
    methods: list[str],
) -> tuple[list[object], list[str]]:
    handles: list[object] = []
    labels: list[str] = []
    for method in methods:
        subdf = grouped[grouped['method'] == method]
        style = METHOD_STYLE.get(method, {'linestyle': '-', 'marker': 'o', 'linewidth': 2.0})
        style_main = dict(style)
        markersize_main = float(style_main.pop('markersize', 6.0))
        x_vals = subdf[x_col].to_numpy(dtype=float)
        mean_vals = subdf['mean'].to_numpy(dtype=float)
        low_vals = subdf['ci_low'].to_numpy(dtype=float)
        high_vals = subdf['ci_high'].to_numpy(dtype=float)
        line, = ax.plot(x_vals, mean_vals, label=_method_label(method), markersize=markersize_main, **style_main)
        ax.fill_between(x_vals, low_vals, high_vals, alpha=0.18, color=style_main.get('color'))
        handles.append(line)
        labels.append(_method_label(method))
    return handles, labels


def _plot_joint_log_time_curves(df: pd.DataFrame, out_path: Path) -> None:
    prepared_k = _prepare_grouped_curve_data(
        df,
        x_col='K_total',
        y_col='solve_time_sec',
        sweep_axis='K',
    )
    prepared_n = _prepare_grouped_curve_data(
        df,
        x_col='N_per_dist',
        y_col='solve_time_sec',
        sweep_axis='N',
    )
    if prepared_k is None or prepared_n is None:
        return

    grouped_k, x_unique_k, methods_k = prepared_k
    grouped_n, x_unique_n, methods_n = prepared_n

    fig, axes = plt.subplots(1, 2, figsize=JOINT_LOG_FIGSIZE)
    ax_k, ax_n = axes

    handles, labels = _plot_grouped_lines(ax_k, grouped=grouped_k, x_col='K_total', methods=methods_k)
    _set_discrete_x_ticks(ax_k, x_unique_k, integer_stride=2)
    ax_k.set_xlabel(r"Number of distributions $K$", fontsize=JOINT_LOG_AXIS_LABELSIZE)
    ax_k.set_ylabel('Solve time (s)', fontsize=JOINT_LOG_AXIS_LABELSIZE)
    ax_k.set_yscale('log')
    _apply_main_xlim(ax_k, 'time_vs_log_joint_left')
    _apply_main_ylim(ax_k, 'time_vs_log_joint_left')
    ax_k.tick_params(axis='both', labelsize=JOINT_LOG_TICK_LABELSIZE)
    ax_k.grid(alpha=JOINT_LOG_GRID_ALPHA, which='both')

    _plot_grouped_lines(ax_n, grouped=grouped_n, x_col='N_per_dist', methods=methods_n)
    _set_discrete_x_ticks(ax_n, x_unique_n)
    ax_n.set_xlabel(r"Samples per distribution $N_k$", fontsize=JOINT_LOG_AXIS_LABELSIZE)
    ax_n.set_ylabel('', fontsize=JOINT_LOG_AXIS_LABELSIZE)
    ax_n.set_yscale('log')
    _apply_main_xlim(ax_n, 'time_vs_log_joint_right')
    _apply_main_ylim(ax_n, 'time_vs_log_joint_right')
    ax_n.tick_params(axis='both', labelsize=JOINT_LOG_TICK_LABELSIZE)
    ax_n.grid(alpha=JOINT_LOG_GRID_ALPHA, which='both')

    legend_kwargs = {
        'frameon': True,
        'fontsize': JOINT_LOG_LEGEND_FONTSIZE,
        'fancybox': False,
        'framealpha': 0.8,
        'loc': 'upper center',
        'ncol': max(1, len(labels)),
        'borderpad': JOINT_LOG_LEGEND_BORDERPAD,
        'handletextpad': JOINT_LOG_LEGEND_HANDLETEXTPAD,
        'columnspacing': JOINT_LOG_LEGEND_COLUMNSPACING,
        'labelspacing': JOINT_LOG_LEGEND_LABELSPACING,
    }
    manual_legend = MANUAL_LEGEND_KWARGS.get('time_vs_log_joint')
    if manual_legend:
        legend_kwargs.update(manual_legend)
        if 'bbox_to_anchor' in manual_legend and 'bbox_transform' not in manual_legend:
            legend_kwargs['bbox_transform'] = fig.transFigure
    legend = fig.legend(handles, labels, **legend_kwargs)
    legend.get_frame().set_edgecolor('0.3')
    legend.get_frame().set_linewidth(0.5)

    fig.tight_layout(rect=JOINT_LOG_LAYOUT_RECT, w_pad=JOINT_LOG_W_PAD)
    fig.savefig(out_path, bbox_inches='tight', bbox_extra_artists=(legend,), pad_inches=0.1)
    plt.close(fig)


def _plot_joint_objective_curves(df: pd.DataFrame, out_path: Path) -> None:
    prepared_k = _prepare_grouped_curve_data(
        df,
        x_col='K_total',
        y_col='objective',
        sweep_axis='K',
    )
    prepared_n = _prepare_grouped_curve_data(
        df,
        x_col='N_per_dist',
        y_col='objective',
        sweep_axis='N',
    )
    if prepared_k is None or prepared_n is None:
        return

    grouped_k, x_unique_k, methods_k = prepared_k
    grouped_n, x_unique_n, methods_n = prepared_n

    fig, axes = plt.subplots(1, 2, figsize=JOINT_OBJECTIVE_FIGSIZE)
    ax_k, ax_n = axes

    handles, labels = _plot_grouped_lines(ax_k, grouped=grouped_k, x_col='K_total', methods=methods_k)
    _set_discrete_x_ticks(ax_k, x_unique_k, integer_stride=2)
    ax_k.set_xlabel(r"Number of distributions $K$", fontsize=JOINT_OBJECTIVE_AXIS_LABELSIZE)
    ax_k.set_ylabel('Robust objective value', fontsize=JOINT_OBJECTIVE_AXIS_LABELSIZE)
    _apply_main_xlim(ax_k, 'objective_vs_joint_left')
    _apply_main_ylim(ax_k, 'objective_vs_joint_left')
    ax_k.tick_params(axis='both', labelsize=JOINT_OBJECTIVE_TICK_LABELSIZE)
    ax_k.grid(alpha=JOINT_OBJECTIVE_GRID_ALPHA)

    _plot_grouped_lines(ax_n, grouped=grouped_n, x_col='N_per_dist', methods=methods_n)
    _set_discrete_x_ticks(ax_n, x_unique_n)
    ax_n.set_xlabel(r"Samples per distribution $N_k$", fontsize=JOINT_OBJECTIVE_AXIS_LABELSIZE)
    ax_n.set_ylabel('', fontsize=JOINT_OBJECTIVE_AXIS_LABELSIZE)
    _apply_main_xlim(ax_n, 'objective_vs_joint_right')
    _apply_main_ylim(ax_n, 'objective_vs_joint_right')
    ax_n.tick_params(axis='both', labelsize=JOINT_OBJECTIVE_TICK_LABELSIZE)
    ax_n.grid(alpha=JOINT_OBJECTIVE_GRID_ALPHA)

    ylims_k = ax_k.get_ylim()
    ylims_n = ax_n.get_ylim()
    if np.allclose(ylims_k, ylims_n, rtol=0.0, atol=1e-12):
        ax_n.tick_params(axis='y', labelleft=False)

    legend_kwargs = {
        'frameon': True,
        'fontsize': JOINT_OBJECTIVE_LEGEND_FONTSIZE,
        'fancybox': False,
        'framealpha': 0.8,
        'loc': 'upper center',
        'ncol': max(1, len(labels)),
        'borderpad': JOINT_OBJECTIVE_LEGEND_BORDERPAD,
        'handletextpad': JOINT_OBJECTIVE_LEGEND_HANDLETEXTPAD,
        'columnspacing': JOINT_OBJECTIVE_LEGEND_COLUMNSPACING,
        'labelspacing': JOINT_OBJECTIVE_LEGEND_LABELSPACING,
    }
    manual_legend = MANUAL_LEGEND_KWARGS.get('objective_vs_joint')
    if manual_legend:
        legend_kwargs.update(manual_legend)
        if 'bbox_to_anchor' in manual_legend and 'bbox_transform' not in manual_legend:
            legend_kwargs['bbox_transform'] = fig.transFigure
    legend = fig.legend(handles, labels, **legend_kwargs)
    legend.get_frame().set_edgecolor('0.3')
    legend.get_frame().set_linewidth(0.5)

    fig.tight_layout(rect=JOINT_OBJECTIVE_LAYOUT_RECT, w_pad=JOINT_OBJECTIVE_W_PAD)
    fig.savefig(out_path, bbox_inches='tight', bbox_extra_artists=(legend,), pad_inches=0.1)
    plt.close(fig)


def plot_scaling(df: pd.DataFrame, out_dir: str | Path, *, timestamp_source: str | Path | None = None) -> None:
    _configure_plot_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    df = df.copy()
    df = df[df['method'].isin(METHOD_ORDER)]
    match = re.search(r'\d{8}_\d{6}', Path(timestamp_source or out_dir).name)
    suffix = '_' + match.group(0) if match else ''
    _plot_joint_log_time_curves(df, out_dir / f'time_vs_log_joint{suffix}.pdf')
    _plot_joint_objective_curves(df, out_dir / f'objective_vs_joint{suffix}.pdf')
