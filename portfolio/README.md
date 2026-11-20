# Portfolio Optimization: Section 5.1

This experiment compares the exact intersected formulation with affine, static, mixture, and Minkowski approximations. It reports solve times and in-sample robust objective values for a long-only portfolio of ten assets.


Running the command below with the default settings reproduces the reported objective values.

## Experimental Setup

The loss combines negative expected return and CVaR with `rho = 10` and `eta = 0.2`. Each reference distribution is Gaussian.

| Parameter | Value in Section 5.1 |
| --- | --- |
| Number of assets | 10 |
| Active mean block | 4 assets; consecutive blocks overlap in 2 assets |
| Mean construction | Combinatorial overlapping blocks |
| Initial high/low means | 0.015 / -0.005 |
| Per-reference high growth / low decrease | 0.001 / 0.001 |
| Systematic / idiosyncratic standard deviations | 0.02 / 0.015 |
| Covariance | Reference-specific random unit factor direction, redrawn in each repetition |
| Reference weights | Uniform |
| Radius multiplier delta | 0.01 |
| K sweep | K = 2, ..., 20 with 10 samples per reference |
| Sample-size sweep | N_k = 5, 10, 20, ..., 100 with K = 3 |
| Repetitions | 10 |

For each reference, the radius is `epsilon_k = (1 + delta) * W2(P_k, P_gelb)`, where `P_gelb` is the sampled Gelbrich-Gaussian center. In the K sweep, this center is constructed once from all 20 references and held fixed. Exact, affine, static, and Minkowski use the individual radii; mixture uses the single-ball radius `(sum_k t_k epsilon_k^2)^(1/2)`, which guarantees that the mixture ball contains the Minkowski sum of the balls.

The sampled center has N_k support points. Its Gaussian sampling seed is fixed at `seed_base`, while the empirical reference samples and covariance directions change across repetitions. The block of high-mean assets of each reference is chosen deterministically. The first block is {1, 2, 3, 4}. Each further block is an unused four-asset block that shares exactly two assets with the previous block; among these, the block whose assets were used most often so far is chosen, ties are broken by the largest total overlap with the earlier blocks, and remaining ties by lexicographic order.

Run this fixed-center experiment with `scaling`.

## Run

After the root installation steps, run this command from the `fusion-DRO` directory:

```bash
python portfolio/main.py scaling \
  --data synthetic --wass-order 2 --d-assets 10 \
  --K-grid 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --N-list 5,10,20,30,40,50,60,70,80,90,100 \
  --K-for-N-sweep 3 --N-for-K-sweep 10 \
  --num-runs 10 --seed-base 0 \
  --mean-pattern combinatorial_overlap_blocks --block-size 4 --stride 2 \
  --mu-high 0.015 --mu-low -0.005 \
  --mu-high-growth 0.001 --mu-low-shrink 0.001 \
  --cov-pattern heterogeneous --cov-direction-seed 12345 --cov-direction-per-run \
  --sigma-sys 0.02 --sigma-id 0.015 \
  --rho 10 --eta 0.2 \
  --fixed-delta 0.01 \
  --run-exact True --run-mixture True --run-minkowski True \
  --approx-mode both \
  --max-exact-constraints 35000 --exact-time-limit-sec none \
  --solver MOSEK --solver-threads 4
```

Reference weights are uniform. Repetition r uses empirical-sample seed `seed_base + r` and covariance-direction seed `cov_direction_seed + seed_base + r`, for r = 0, ..., 9. The exact reformulation has 2 + 4 ∏ₖ Nₖ constraints. In our runs, instances above 35,000 constraints, starting with K = 4 and N_k = 10, ran out of memory. The default caps (`--max-exact-constraints 35000`, `--max-tuples 200000`) therefore skip these instances and record them as `skipped_constraints_…` or `skipped_tuples_…` in `scaling.csv`, instead of letting the run fail. They are the missing points of the exact curves in Figures 1–2. Pass `none` to either cap to attempt them on a machine with more memory. The exact solves run without a time limit. If MOSEK stops without certifying optimality, its last solution is reported with status `optimal_inaccurate`. Every MOSEK solve uses 4 threads, so the results do not depend on the number of cores of the machine. All settings in the command above are also defaults.

## Results

Each run writes `config.json` and `scaling.csv` to `portfolio/results/res_w2_<timestamp>_syn/`. The two joint figures are:

- `time_vs_log_joint_<timestamp>.pdf`: mean solve times with 95% confidence intervals, on a logarithmic scale (Figure 1).
- `objective_vs_joint_<timestamp>.pdf`: mean robust objective values with 95% confidence intervals (Figure 2).

Regenerate the figures from saved results:

```bash
python portfolio/main.py replot \
  --results-dir portfolio/results/res_w2_<timestamp>_syn
```

Use `--out-dir /path/to/figures` with `replot` to leave the original result directory unchanged.

### Worst-case distributions

The worst-case transport figure is produced by re-solving one instance of a scaling run, here run 0 of the sample-size sweep with K = 3 and N_k = 10, and plotting the worst-case distributions recovered from the dual solutions:

```bash
python -m portfolio.plotting.worst_case_duals \
  --results portfolio/results/res_w2_<timestamp>_syn --run 0 --sweep N --K 3 --N 10 --solver MOSEK
python -m portfolio.plotting.plot_worst_case_contours \
  --npz portfolio/results/res_w2_<timestamp>_syn/worst_case/worst_case_run0_N_K3_N10.npz
```

The first command writes the recovered worst cases to `worst_case/worst_case_run0_N_K3_N10.npz`, together with a summary of the objective values. The second writes `worst_case_run0_N_K3_N10_contours_2x3.pdf` to the same folder. The plot uses LaTeX for its text if available.

## Checks

From `fusion-DRO`, run `python -m unittest portfolio.test_scaling`. These checks cover the paper defaults, deterministic reference prefixes, covariance resampling, retry units, and figure output names; they do not require a solver license.
