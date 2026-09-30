"""Worst-case transport of the training samples, one panel per method.

Reads the .npz written by worst_case_duals.py and draws, for exact, affine, static,
Minkowski and mixture:

  x      = return of the Exact portfolio, beta^T xi (the far-left tail is compressed).
  y      = xi^T e, where e is the first principal component of the reference sample means,
           projected onto the directions orthogonal to the Exact portfolio.
  colour = in-sample objective max_i {b_i beta_m^T xi + c_i tau_m} of the panel's own
           decision (beta_m, tau_m), on its own colour scale; its contour lines are the
           lines along which beta_m^T xi is constant. Dashed line: Exact VaR (return = -tau).
  circles  = training samples, coloured by reference; arrows = worst-case transport
             (opacity = mass); diamonds = worst-case locations; black open circles =
             Exact's worst-case support points (area = mass).
  top      = distribution of each reference along x before (filled) and after (line)
             the transport; box = pairwise W2 between the transported references Q_k.

Run from fusion-DRO/:
  python -m portfolio.plotting.plot_worst_case_contours --npz <file.npz>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrow, Patch
from matplotlib.ticker import MaxNLocator
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

METHODS = [("exact", "Exact"), ("affine", "Affine"), ("static", "Static"), ("minkowski", "Minkowski"), ("mixture", "Mixture")]
REF_COLORS = ["#3f9c6d", "#d9709c", "#7d5ba6", "#e0a458", "#4f86c6", "#6b6b6b"]   # green, pink, purple, ...
EDGE = "#262626"


def configure_plot_style(use_latex: bool = True) -> None:
    """Text style of the paper figures (e.g. images/objective_vs_joint_*.pdf): Times text via
    \\usepackage{times} and Computer Modern math, rendered by LaTeX.

    Note: rcParams font.serif = ["Times"] makes Matplotlib add \\usepackage{mathptmx}, which switches
    the *math* to Times as well; listing a serif font Matplotlib does not map to a LaTeX package
    avoids that, so only the text is Times and the math stays Computer Modern."""
    plt.rcParams.update({"xtick.major.size": 6, "ytick.major.size": 6, "xtick.major.width": 1, "ytick.major.width": 1,
                         "mathtext.fontset": "cm"})
    if use_latex and shutil.which("latex") is not None:
        plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Nimbus Roman No9 L"]})
        plt.rcParams["text.latex.preamble"] = r"""\usepackage{amsmath,amsfonts,amssymb,bm} \usepackage{times}"""
    else:
        plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                             "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"]})
        plt.rcParams["text.latex.preamble"] = ""


def soft_cmap(name: str, lighten: float = 0.45, trim: float = 0.06):
    """A softer version of a matplotlib colormap: extremes trimmed and mixed with white."""
    base = plt.get_cmap(name)
    cols = base(np.linspace(trim, 1.0 - trim, 256))
    cols[:, :3] = (1.0 - lighten) * cols[:, :3] + lighten
    return LinearSegmentedColormap.from_list(f"soft_{name}", cols)


def w2(P, p, Q, q):
    """2-Wasserstein distance between two discrete distributions (transport LP)."""
    p, q = p / p.sum(), q / q.sum()
    C = ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
    n1, n2 = C.shape
    ii, jj = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    var = (ii * n2 + jj).ravel()
    A = coo_matrix((np.ones(2 * n1 * n2), (np.concatenate([ii.ravel(), n1 + jj.ravel()]), np.concatenate([var, var]))),
                   shape=(n1 + n2, n1 * n2)).tocsr()
    r = linprog(C.ravel(), A_eq=A[:-1], b_eq=np.concatenate([p, q])[:-1], bounds=(0, None), method="highs")
    if r.fun is None:
        raise RuntimeError(f"W2 transport LP failed: {r.message}")
    return float(np.sqrt(max(r.fun, 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--mass_tol", type=float, default=1e-3, help="ignore transports carrying less than this share of a sample")
    ap.add_argument("--layout", choices=["2x3", "1x5"], default="2x3")
    ap.add_argument("--cmap", default="RdBu_r", help="colormap (softened); first colour = lowest objective")
    ap.add_argument("--lighten", type=float, default=0.35, help="how much the colormap is mixed with white (0 = original)")
    ap.add_argument("--ghost_exact", dest="ghost_exact", action="store_true",
                    help="repeat the Exact worst-case points (dashed) in the other panels (off by default)")
    ap.add_argument("--xaxis", choices=["return", "objective"], default="return",
                    help="x = return of the Exact portfolio (samples spread out; default) or its objective integrand")
    ap.add_argument("--tail_share", type=float, default=0.35,
                    help="(x = return) width of the compressed far-left piece relative to the sample range")
    ap.add_argument("--color_scale", choices=["panel", "equalized", "log"], default="panel",
                    help="panel: each panel coloured from its own lowest (blue) to highest (red) objective, spread "
                         "evenly over its area; equalized: the same, but one scale shared by all panels; log: shared log scale")
    ap.add_argument("--no_marginals", dest="marginals", action="store_false",
                    help="do not draw the distributions of the references along x above each panel")
    ap.add_argument("--kde_bw", type=float, default=0.03,
                    help="bandwidth of the marginal densities, as a share of the x-axis width")
    ap.add_argument("--out", default="")
    ap.add_argument("--no_latex", action="store_true", help="do not use LaTeX for text (Times-like fonts instead)")
    a = ap.parse_args()
    for use_latex in ([False] if a.no_latex else [True, False]):
        try:
            render(a, use_latex)
            return
        except RuntimeError as exc:          # e.g. a LaTeX package missing on this machine
            if "latex" not in str(exc).lower() or not use_latex:
                raise
            plt.close("all")
            print("[plot] LaTeX rendering failed; falling back to Matplotlib text (Times-like fonts).", flush=True)


def render(a, use_latex: bool) -> None:
    configure_plot_style(use_latex)
    tex = bool(plt.rcParams["text.usetex"])
    SC = (lambda w: rf"\textsc{{{w}}}") if tex else (lambda w: w)                     # method names in small caps
    BM = (lambda w: rf"\bm{{\{w}}}" if len(w) > 1 else rf"\bm{{{w}}}") if tex else \
        (lambda w: rf"\boldsymbol{{\{w}}}" if len(w) > 1 else rf"\mathbf{{{w}}}")      # bold symbols
    cmap = soft_cmap(a.cmap, lighten=a.lighten)

    z = np.load(a.npz)
    S = z["samples"]                      # (K, N, m)
    K, N, m = S.shape
    t = z["t"]
    rho, eta = float(z["rho"]), float(z["eta"])

    def f(meth, X):
        beta, tau = z[f"{meth}__beta"], float(z[f"{meth}__tau"])
        r = np.asarray(X, float) @ beta
        return -r + rho * (tau + np.maximum(-r - tau, 0.0) / eta)

    # ---- axes
    be, tau_e = z["exact__beta"], float(z["exact__tau"])
    xbar = S.reshape(-1, m).mean(0)
    e1 = be / np.linalg.norm(be)
    Mc = S.mean(1) - S.mean(1).mean(0)
    Mc = Mc - np.outer(Mc @ e1, e1)
    e2 = np.linalg.svd(Mc, full_matrices=False)[2][0]
    e2 = e2 - (e2 @ e1) * e1
    e2 /= np.linalg.norm(e2)
    f_kink = tau_e * (1.0 + rho)          # f_exact at the Exact VaR (return = -tau)

    def coords(X):
        """(raw x, y): raw x is the Exact return or f_exact, depending on --xaxis."""
        X = np.atleast_2d(np.asarray(X, float))
        rx = X @ be if a.xaxis == "return" else f("exact", X)
        return np.stack([rx, X @ e2], axis=-1)

    def return_from_fexact(fx):
        fx = np.asarray(fx, float)
        right = rho * tau_e - fx                                              # piece with return >= -tau
        left = (rho * tau_e * (1.0 - 1.0 / eta) - fx) / (1.0 + rho / eta)     # piece with return <  -tau
        return np.where(fx <= f_kink, right, left)

    def lift(rx, y):
        r = np.asarray(rx, float) if a.xaxis == "return" else return_from_fexact(rx)
        return xbar[None, :] + np.outer((r - xbar @ be) / (be @ be), be) + np.outer(np.asarray(y, float) - xbar @ e2, e2)

    # ---- transports per method: (origin, destination, share of the sample's mass, reference)
    def transports(meth):
        out = []
        if meth == "exact":
            al, pts, ms = z["exact__alphas"], z["exact__points"], z["exact__mass"]
            for ai in range(len(al)):
                for l in range(2):
                    if ms[ai, l] * N >= a.mass_tol:
                        for k in range(K):
                            out.append((S[k, al[ai, k]], pts[ai, l], ms[ai, l] * N, k))
            return out
        mv, ms = z[f"{meth}__moved"], z[f"{meth}__mass"]
        for k in range(K):
            for i in range(N):
                for l in range(2):
                    w = ms[k, i, l] * (N / t[k] if meth == "mixture" else N)
                    if w >= a.mass_tol:
                        out.append((S[k, i], mv[k, i, l], w, k))
        return out

    T = {meth: transports(meth) for meth, _ in METHODS}

    # ---- W2 distance of each worst case to the Exact one
    ex_pts, ex_w = z["exact__points"].reshape(-1, m), z["exact__mass"].reshape(-1)
    keep = ex_w > 1e-5 * ex_w.sum()                         # smaller masses are solver noise
    ex_pts, ex_w = ex_pts[keep], ex_w[keep]
    dist = {}
    for meth, _ in METHODS[1:]:
        mv, ms = z[f"{meth}__moved"], z[f"{meth}__mass"]
        P = np.concatenate([mv[k].reshape(-1, m) for k in range(K)])
        wts = np.concatenate([(ms[k] / ms[k].sum() * t[k]).reshape(-1) for k in range(K)])
        kk = wts > 1e-5
        dist[meth] = w2(P[kk], wts[kk], ex_pts, ex_w)

    # ---- pairwise W2 between the references, before and after each method's worst-case transport
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    w2_before = {(i, j): w2(S[i], np.ones(N), S[j], np.ones(N)) for i, j in pairs}

    def ref_after(meth, k):
        P = np.array([d for (_, d, w, kk) in T[meth] if kk == k])
        wts = np.array([w for (_, d, w, kk) in T[meth] if kk == k])
        return P, wts

    w2_after = {}
    for meth, _ in METHODS:
        Rk = [ref_after(meth, k) for k in range(K)]
        w2_after[meth] = {(i, j): w2(Rk[i][0], Rk[i][1], Rk[j][0], Rk[j][1]) for i, j in pairs}
    print("W2 between references before:", {f"{i+1}{j+1}": round(v, 4) for (i, j), v in w2_before.items()})
    for meth, _ in METHODS:
        print(f"W2 between references after {meth}:", {f"{i+1}{j+1}": round(v, 4) for (i, j), v in w2_after[meth].items()})

    # ---- grid, colour scale
    cS_raw = coords(S.reshape(-1, m))
    allc = np.vstack([cS_raw] + [coords(np.array([d for _, d, _, _ in T[mm]])) for mm, _ in METHODS if T[mm]])
    ylo, yhi = allc[:, 1].min(), allc[:, 1].max()
    # display transform of x: return -> true scale over the samples, compressed piece for the
    # far-left worst-case points; objective -> log scale
    if a.xaxis == "return":
        xa = cS_raw[:, 0].min() - 0.15 * np.ptp(cS_raw[:, 0])
        xmin = allc[:, 0].min()
        lam = min(1.0, a.tail_share * (cS_raw[:, 0].max() - xa) / (xa - xmin)) if xmin < xa else 1.0
        disp = lambda x: np.where(np.asarray(x, float) >= xa, np.asarray(x, float), xa - lam * (xa - np.asarray(x, float)))
        undisp = lambda u: np.where(np.asarray(u, float) >= xa, np.asarray(u, float), xa - (xa - np.asarray(u, float)) / lam)
    else:
        xa, lam = None, 1.0
        disp = lambda x: np.log10(np.asarray(x, float))
        undisp = lambda u: 10.0 ** np.asarray(u, float)
    # axis limits sit on the first and last tick, with every point at least 2% inside
    ymarg = 0.02 * (yhi - ylo)
    yticks = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(ylo - ymarg, yhi + ymarg)
    yticks = yticks[(yticks >= yticks[yticks <= ylo - ymarg].max()) & (yticks <= yticks[yticks >= yhi + ymarg].min())]
    if a.xaxis == "return":
        step = 0.005
        cloud_w = float(np.ptp(cS_raw[:, 0]))
        x_right = np.ceil((allc[:, 0].max() + 0.02 * cloud_w) / step) * step
        x_left = np.floor((allc[:, 0].min() - 0.02 * cloud_w / max(lam, 1e-9)) / 0.01) * 0.01   # 2% margin in display units
        cloud = np.arange(np.ceil(xa / step) * step, x_right + step / 2, step) if lam < 1.0 else \
            np.arange(x_left, x_right + step / 2, step)
        xticks = cloud
        if lam < 1.0:
            gap = 0.9 * (disp(cloud[1]) - disp(cloud[0]))   # keep tail tick labels as far apart as the others
            tail = [x_left]
            for v in np.arange(x_left + 0.01, xa, 0.01):
                if disp(v) - disp(tail[-1]) > gap and disp(cloud[0]) - disp(v) > gap:
                    tail.append(v)
            xticks = np.concatenate([tail, cloud])
    else:
        xticks = np.array([v for v in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5)
                           if allc[:, 0].min() / 2.5 <= v <= allc[:, 0].max() * 2.5])
        xticks = xticks[(xticks >= xticks[xticks <= allc[:, 0].min()].max()) & (xticks <= xticks[xticks >= allc[:, 0].max()].min())]
    ulo, uhi = float(disp(xticks[0])), float(disp(xticks[-1]))
    gyv = np.linspace(yticks[0], yticks[-1], 300)
    gu, gy = np.meshgrid(np.linspace(ulo, uhi, 500), gyv)
    grid = lift(undisp(gu.ravel()), gy.ravel())
    F = {meth: f(meth, grid).reshape(gu.shape) for meth, _ in METHODS}
    allF = np.concatenate([v.ravel() for v in F.values()])
    vmin, vmax = float(allF.min()), float(allF.max())
    from matplotlib.colors import FuncNorm, Normalize

    def equalizing_norm(values):
        # colour position = share of the plotted area with a smaller objective value, so the
        # colours change evenly across the area (continuous, no bands)
        qs = np.linspace(0.0, 1.0, 201)
        qv = np.maximum.accumulate(np.quantile(values, qs)) + np.arange(qs.size) * 1e-12
        return FuncNorm((lambda v: np.interp(v, qv, qs), lambda q: np.interp(q, qs, qv)),
                        vmin=float(values.min()), vmax=float(values.max()))

    if a.color_scale == "panel":
        norms = {meth: equalizing_norm(F[meth].ravel()) for meth, _ in METHODS}
        levels = {meth: np.quantile(F[meth], np.linspace(0.1, 0.9, 9)) for meth, _ in METHODS}
    elif a.color_scale == "equalized":
        shared = equalizing_norm(allF)
        norms = {meth: shared for meth, _ in METHODS}
        levels = {meth: np.quantile(allF, np.linspace(0.1, 0.9, 9)) for meth, _ in METHODS}
    else:
        shared = LogNorm(vmin=vmin, vmax=vmax) if vmin > 0 else SymLogNorm(linthresh=0.05, vmin=vmin, vmax=vmax)
        norms = {meth: shared for meth, _ in METHODS}
        lv = np.geomspace(vmin, vmax, 9)[1:-1] if vmin > 0 else np.linspace(vmin, vmax, 9)[1:-1]
        levels = {meth: lv for meth, _ in METHODS}
    D = lambda c: np.stack([disp(c[..., 0]), c[..., 1]], axis=-1)   # raw coords -> display coords

    # ---- figure
    # the marginal strips are inset axes above each panel, which constrained_layout ignores,
    # so the spacing is set by hand when they are drawn
    strip_h = 0.36 if a.marginals else 0.0            # strip height, as a share of the panel height
    if a.layout == "2x3":
        fig, axes = plt.subplots(2, 3, figsize=(18, 11 + 3.2 * strip_h), constrained_layout=not a.marginals)
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 6, figsize=(29, 5.4 * (1 + strip_h)), constrained_layout=not a.marginals,
                                 gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 0.5]})
    if a.marginals:
        fig.subplots_adjust(left=0.05, right=0.965, bottom=0.06, top=0.87, wspace=0.26,
                            hspace=0.27 + 1.3 * strip_h if a.layout == "2x3" else 0.2)
    cS = [D(coords(S[k])) for k in range(K)]
    meshes = {}
    for ax, (meth, title) in zip(axes, METHODS):
        mesh = ax.pcolormesh(gu, gy, F[meth], cmap=cmap, norm=norms[meth], shading="gouraud", rasterized=True, zorder=0)
        meshes[meth] = mesh
        ax.contour(gu, gy, F[meth], levels=levels[meth], colors="k", linewidths=0.4, alpha=0.25, zorder=1)
        kink_raw = -tau_e if a.xaxis == "return" else f_kink
        ax.axvline(float(disp(kink_raw)), color="k", ls="--", lw=1.0, alpha=0.6, zorder=2)
        if meth != "exact" and a.ghost_exact:
            ms_ = z["exact__mass"].reshape(-1)
            k_ = ms_ * N >= a.mass_tol
            q = D(coords(z["exact__points"].reshape(-1, m)[k_]))
            ax.scatter(q[:, 0], q[:, 1], s=25 + 1200 * ms_[k_], facecolor="none", edgecolor=EDGE,
                       linewidth=1.0, linestyle="--", alpha=0.75, zorder=3, clip_on=False)
        for (o, dst, w, k) in T[meth]:
            po, pd_ = D(coords(o))[0], D(coords(dst))[0]
            if np.allclose(po, pd_, rtol=1e-3, atol=1e-6):
                continue
            ax.annotate("", xy=pd_, xytext=po, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-|>", color=REF_COLORS[k], lw=1.0, mutation_scale=9,
                                        alpha=float(np.clip(0.2 + 0.8 * w, 0, 1)), shrinkA=3, shrinkB=3), zorder=4)
        if meth == "exact":
            ms_ = z["exact__mass"].reshape(-1)
            k_ = ms_ * N >= a.mass_tol
            q = D(coords(z["exact__points"].reshape(-1, m)[k_]))
            ax.scatter(q[:, 0], q[:, 1], s=25 + 1200 * ms_[k_], facecolor="none", edgecolor="k", linewidth=1.4, zorder=5,
                       clip_on=False)
        else:
            for (_, dst, w, k) in T[meth]:
                pp = D(coords(dst))[0]
                ax.scatter(pp[0], pp[1], marker="D", s=30, color=REF_COLORS[k], edgecolor=EDGE, linewidth=0.6,
                           alpha=float(np.clip(0.3 + 0.7 * w, 0, 1)), zorder=5, clip_on=False)
        for k in range(K):
            ax.scatter(cS[k][:, 0], cS[k][:, 1], marker="o", s=40, color=REF_COLORS[k], edgecolor=EDGE, linewidth=0.7,
                       zorder=6, clip_on=False)
        if not a.marginals:
            ax.set_title(SC(title), fontsize=26)
        txt = "\n".join(rf"$\mathsf{{W}}_2(\mathbb Q_{i + 1},\mathbb Q_{j + 1})={w2_after[meth][(i, j)]:.3f}$" for i, j in pairs)
        ax.text(0.025, 0.975, txt, transform=ax.transAxes, ha="left", va="top", fontsize=16, zorder=8,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#9a9a9a", alpha=0.85))
        ax.set_xlim(ulo, uhi)
        ax.set_ylim(yticks[0], yticks[-1])
        ax.set_xticks(disp(xticks))
        ax.set_xticklabels([("$0$" if abs(v) < 1e-12 else f"${v:g}$") for v in xticks])
        ax.set_yticks(yticks)
        ax.set_yticklabels([("$0$" if abs(v) < 1e-12 else f"${v:g}$") for v in yticks])
        if a.xaxis == "return":
            ax.set_xlabel(rf"Return of the {SC('Exact')} portfolio (${BM('beta')}^\top{BM('xi')}$)", fontsize=21)
        else:
            ax.set_xlabel(rf"In-sample objective under the {SC('Exact')} decision", fontsize=21)
        # the y-axis is shared by all panels: label and tick values only in the first column
        col = list(axes).index(ax) % (3 if a.layout == "2x3" else 6)
        if col == 0 or a.layout != "2x3":
            ax.set_ylabel(rf"First principal component (${BM('xi')}^\top\boldsymbol{{e}}$)", fontsize=21)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=17)

    # ---- marginal distributions along x: for each reference, before (filled) and after
    # (line) the method's worst-case transport; one row per reference, as a ridgeline
    if a.marginals:
        uu = np.linspace(ulo, uhi, 600)
        bw = a.kde_bw * (uhi - ulo)

        def kde(xs, ws):
            xs, ws = np.asarray(xs, float), np.asarray(ws, float)
            ws = ws / ws.sum()
            return (ws[None, :] * np.exp(-0.5 * ((uu[:, None] - xs[None, :]) / bw) ** 2)).sum(1) / (bw * np.sqrt(2 * np.pi))

        before = [kde(cS[k][:, 0], np.ones(N)) for k in range(K)]
        after_all = {meth: [kde(*zip(*[(D(coords(dst))[0][0], w) for (_, dst, w, kk) in T[meth] if kk == k]))
                            for k in range(K)] for meth, _ in METHODS}
        # one vertical scale per reference, shared by all panels, so the reference distribution
        # (filled) is drawn identically everywhere and the transported one (line) is comparable
        row_scale = [0.9 / max([before[k].max()] + [after_all[mm][k].max() for mm, _ in METHODS]) for k in range(K)]
        for ax, (meth, title) in zip(axes, METHODS):
            sax = ax.inset_axes([0.0, 1.02, 1.0, strip_h])
            sax.set_xlim(ulo, uhi)
            for k in range(K):
                after = after_all[meth][k]
                scale = row_scale[k]
                base = K - 1 - k                                         # reference 1 on top
                sax.fill_between(uu, base, base + scale * before[k], color=REF_COLORS[k], alpha=0.30, lw=0, zorder=2 + k)
                sax.plot(uu, base + scale * before[k], color=REF_COLORS[k], lw=0.6, alpha=0.6, zorder=2 + k)
                sax.plot(uu, base + scale * after, color=REF_COLORS[k], lw=1.8, zorder=5 + k)
                sax.axhline(base, color="#bdbdbd", lw=0.5, zorder=1)
            sax.axvline(float(disp(-tau_e if a.xaxis == "return" else f_kink)), color="k", ls="--", lw=1.0, alpha=0.6)
            sax.set_ylim(-0.05, K - 0.05)
            sax.set_xticks([])
            sax.set_yticks([])
            for sp in ("top", "right", "left"):
                sax.spines[sp].set_visible(False)
            sax.set_title(SC(title), fontsize=26)

    leg_ax = axes[-1]
    leg_ax.axis("off")
    handles = [Line2D([], [], marker="o", color=REF_COLORS[k], markeredgecolor=EDGE, ls="", ms=9,
                      label=rf"Training samples of $\hat{{\mathbb P}}_{k + 1}$") for k in range(K)]
    handles += [Line2D([], [], marker="D", color="#9a9a9a", markeredgecolor=EDGE, ls="", ms=7,
                       label="Worst-case location"),
                Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="k", markeredgewidth=1.4, ls="", ms=12,
                       label=rf"{SC('Exact')} worst case (area = mass)"),
                FancyArrow(0, 0, 1, 0, color="#6b6b6b", label="Transport (opacity = mass)"),
                Line2D([], [], color="k", ls="--", lw=1.0, alpha=0.6, label=rf"{SC('Exact')} VaR")]
    if a.marginals:
        handles += [Patch(facecolor="#9a9a9a", alpha=0.35, edgecolor="none", label=r"Top: reference distributions $\hat{\mathbb P}_k$"),
                    Line2D([], [], color="#6b6b6b", lw=1.8, label=r"Top: transported distributions $\mathbb Q_k$")]
    def _arrow_handle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        y = 0.5 * height - ydescent
        return FancyArrow(-xdescent, y, width, 0, width=0.6, head_width=0.45 * height, head_length=0.3 * height,
                          length_includes_head=True, color=orig_handle.get_facecolor(), lw=1.2)

    leg_ax.legend(handles=handles, handler_map={FancyArrow: HandlerPatch(patch_func=_arrow_handle)}, loc="center", bbox_to_anchor=(0.5, 0.5 + strip_h / 2), frameon=False, fontsize=20,
                  labelspacing=0.45)
    def nice_ticks(nrm, lo, hi, min_gap=0.07):
        """Round values placed where they fall on a (possibly nonlinear) colour scale; ticks closer
        than min_gap of the bar are dropped; both ends are always shown."""
        nice = np.array([0.02, 0.05, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
                         0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 5])
        cand = [lo] + [v for v in nice if lo < v < hi] + [hi]
        pos = [float(nrm(v)) for v in cand]
        out, last = [cand[0]], pos[0]
        for v, ps in zip(cand[1:-1], pos[1:-1]):
            if ps - last >= min_gap and pos[-1] - ps >= min_gap:
                out.append(v)
                last = ps
        return out + [cand[-1]]

    if a.color_scale == "panel":
        # one colour bar per panel: each panel's colours run from its own lowest to highest value
        for ax, (meth, _) in zip(axes, METHODS):
            cbp = fig.colorbar(meshes[meth], ax=ax, fraction=0.05, pad=0.02)
            tv = nice_ticks(norms[meth], float(F[meth].min()), float(F[meth].max()))
            cbp.set_ticks(tv)
            cbp.set_ticklabels([f"${v:.2f}$" for v in tv])
            cbp.ax.tick_params(labelsize=15)
            cbp.set_label("In-sample objective", fontsize=18, labelpad=8)
    else:
        cb = fig.colorbar(mesh, ax=list(axes[:-1]) if a.layout == "1x5" else axes.tolist(), shrink=0.8, pad=0.01)
        if a.color_scale == "equalized":
            tv = nice_ticks(norms[METHODS[0][0]], vmin, vmax, min_gap=0.06)
            cb.set_ticks(tv)
            cb.set_ticklabels([f"${v:.2f}$" for v in tv])
        cb.set_label("In-sample objective", fontsize=16)
        cb.ax.tick_params(labelsize=14)
    print("W2 to the Exact worst case:", {k: round(v, 4) for k, v in dist.items()})

    out = Path(a.out) if a.out else Path(a.npz).with_name(Path(a.npz).stem + f"_contours_{a.layout}")
    fig.savefig(out.with_suffix(".pdf"), dpi=200, bbox_inches="tight", pad_inches=0.08)
    try:
        fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight", pad_inches=0.08)
    except Exception:   # LaTeX PNG export needs dvipng; fall back to converting the PDF
        if shutil.which("pdftoppm"):
            subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile", str(out.with_suffix(".pdf")), str(out)], check=False)
    print(out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
