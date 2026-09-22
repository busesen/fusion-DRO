# Multi-item Newsvendor: Section 5.2

This experiment evaluates Affine, Static, Barycenter, ERM, Pool, Mixture, Minkowski, and RKHS on coffee demand. Each decision specifies order quantities for small, regular, and large cups.

The RKHS benchmark is the method of Li et al. (2026), cited below. It requires a Gurobi license.

## Data and Experimental Setup

Use the [Maven Roasters Coffee Shop Sales dataset](https://mavenanalytics.io/data-playground/coffee-shop-sales). Place its transaction workbook at `coffee-newsvendor/data/Coffee_Shop_Sales.xlsx`, or supply `--raw_xlsx /path/to/Coffee_Shop_Sales.xlsx`.

Preprocessing retains coffee products sold in all three sizes, sums daily sales, and computes weekly average daily demand. It yields 27 observations for each of five products:

| Identifier | Product |
| --- | --- |
| `group_1` | Brazilian |
| `group_2` | Colombian |
| `group_3` | Ethiopian |
| `group_4` | Jamaican |
| `group_5` | Old Time Diner Blend |

The common selling-price vector is the Brazilian price vector `p = [2.2, 3.0, 3.5]`. The costs are `c = [0.4, 1.9, 2.0]`, `h = 0.05 * p`, and `b = 0.5 * p`.

Each product is held out in turn as the target. Two randomly selected target weeks both define its empirical reference distribution and serve for validation; the remaining 25 weeks are reserved for testing. The other four products contribute their 27 weeks, giving K = 5 references. The target reference receives weight 0.5, and each historical reference receives 0.125. These weights are used for Affine, Static, Mixture, Minkowski, and Barycenter.

Hyperparameters minimize the mean newsvendor cost on the two validation weeks. Affine and Static use `epsilon_k = (1 + delta) * W2(P_k, P_gelb)`. Pool, Mixture, and Barycenter tune a single radius epsilon; Minkowski tunes the same epsilon grid and sets `epsilon_k = epsilon` for every reference. RKHS tunes the robustness parameter $nu$ and uses uniform reference weights. We also tested RKHS with the non-uniform weights used by the other weighted benchmarks, but observed little change in performance. Since this constitutes a modification of the original RKHS implementation, we retain uniform weights in the reported results.

Once the target split and radius construction are fixed, the deterministic benchmarks run once per product. Barycenter and RKHS run over five seeds. The Barycenter benchmark samples 128 points from the Gaussian center. The radius-calibration center also uses 128 Monte Carlo points; these are not additional observed weeks.

## Run

Follow the root installation instructions. Run from the `fusion-DRO` directory:

```bash
python coffee-newsvendor/run_coffee_pipeline.py \
  --target_group all --k_start 5 --k_max 5 \
  --wass_p 2 --target_weight 0.5 --val_fraction 0.1 \
  --seed 1234 --num_seeds 5 \
  --affine_static_center gelbrich \
  --w2_mc_samples 128 --barycenter_num_samples 128 \
  --delta_grid '[0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]' \
  --epsilon_grid '[0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]' \
  --run_rkhs False --rkhs_source_weights uniform \
  --nu_grid '[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]' \
  --rkhs_n_J 2000 --rkhs_is_herding False --rkhs_T 50 \
  --rkhs_time_limit 3600 \
  --target_workers 1 --rkhs_parallel_workers 1 \
  --cvxpy_solver MOSEK
```

This command runs the seven conic benchmarks with MOSEK. To include RKHS, change `--run_rkhs False` to `--run_rkhs True` and supply `--gurobi_license_file /path/to/gurobi.lic` (or set `GRB_LICENSE_FILE`). All packages are listed in `requirements.txt`.

## Results

Outputs are saved in `coffee-newsvendor/results/results_<timestamp>/`, with `config.json` recording the invocation. Each product has its own results under `targets/group_*/`.

The combined workbook `final_test/selection_summary_<timestamp>.xlsx` contains the quantities for Table 1: out-of-sample cost and solve time for each method/product. The `mean oos` column is the mean cost on the 25 unseen target weeks. Barycenter and RKHS results are averaged over the five seeds; deterministic methods contribute single values.

Solve times correspond to the selected hyperparameter, averaged across seeds where applicable. The current exporter also includes separately recorded reference-preparation time in the reported total.

To rebuild one target's summary from saved evaluations:

```bash
python coffee-newsvendor/final_test_plot_coffee.py \
  --eval_csv coffee-newsvendor/results/results_<timestamp>/targets/group_1/solver_validation/eval_empirical_all.csv \
  --outdir coffee-newsvendor/results/results_<timestamp>/targets/group_1/final_test
```

## Checks

From `fusion-DRO`, run `python coffee-newsvendor/test_pipeline.py`. These checks cover the opt-in RKHS defaults, five-reference schedule, kernel calculations, and compatibility with the existing RKHS adapter. Set `FUSION_DRO_TEST_GUROBI=1` to include small licensed RKHS regression tests.

## References
* Xianyu Li, Huan Xu, Xiaolin Huang and Chao Shang. [Robust Out-of-Distribution Stochastic Optimization](https://arxiv.org/abs/2604.20147). *arXiv:2604.20147*. 2026.
