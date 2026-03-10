from collections import defaultdict
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def _group_records(records: Iterable[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        key = f"{r.get('method', '')}-{r.get('estimator', '')}-{r.get('optimizer', '')}"
        groups[key].append(r)
    return groups


def _stack_traj(records: List[Dict], key: str) -> np.ndarray:
    arrs = [np.asarray(r[key], dtype=float) for r in records if key in r]
    if len(arrs) == 0:
        return np.empty((0, 0))
    t_min = min(a.shape[0] for a in arrs)
    arrs = [a[:t_min] for a in arrs]
    return np.stack(arrs, axis=0)


def plot_beta_trajectories(records: Iterable[Dict], outfile: str, title: str = "Policy Parameters") -> None:
    groups = _group_records(records)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    for label, recs in groups.items():
        beta = _stack_traj(recs, "beta")
        if beta.size == 0:
            continue
        it = np.arange(beta.shape[1])

        for run in beta:
            ax1.plot(it, run[:, 0], alpha=0.18, linewidth=1.0)
            ax2.plot(it, run[:, 1], alpha=0.18, linewidth=1.0)

        mu = beta.mean(axis=0)
        ax1.plot(it, mu[:, 0], linewidth=2.2, label=label)
        ax2.plot(it, mu[:, 1], linewidth=2.2, label=label)

    ax1.set_ylabel("beta_1")
    ax2.set_ylabel("beta_2")
    ax2.set_xlabel("Iteration")
    ax1.set_ylim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def plot_dual_trajectories(records: Iterable[Dict], outfile: str, title: str = "Dual Variables") -> None:
    groups = _group_records(records)
    fig, ax = plt.subplots(1, 1)

    for label, recs in groups.items():
        if recs and "mu_k" in recs[0]:
            mu_k = _stack_traj(recs, "mu_k")
            if mu_k.size == 0:
                continue
            # Shape: (R, T, K)
            it = np.arange(mu_k.shape[1])
            mu_mean = mu_k.mean(axis=0)  # (T, K)
            mu_avg = mu_mean.mean(axis=1)
            mu_min = mu_mean.min(axis=1)
            mu_max = mu_mean.max(axis=1)
            ax.plot(it, mu_avg, linewidth=2.2, label=f"{label} (mean mu_k)")
            ax.fill_between(it, mu_min, mu_max, alpha=0.15)
        else:
            mu = _stack_traj(recs, "mu")
            if mu.size == 0:
                continue
            it = np.arange(mu.shape[1])
            for run in mu:
                ax.plot(it, run, alpha=0.2, linewidth=1.0)
            ax.plot(it, mu.mean(axis=0), linewidth=2.2, label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Dual")
    ax.legend(frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)
