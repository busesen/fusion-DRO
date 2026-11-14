# Multi-Item Newsvendor: Section 5.2

This experiment evaluates Affine, Static, Minkowski, Mixture, Barycenter, Pool, ERM and RKHS on coffee demand. A coffee product with only five reliable weeks of sales is combined with the long, partly corrupted sales histories of four other products. Each decision specifies how many small, regular and large cups to stock.

The RKHS benchmark is the method of Li et al. (2026), cited below. It requires a Gurobi license.

## Data

Use the [Maven Roasters Coffee Shop Sales dataset](https://mavenanalytics.io/data-playground/coffee-shop-sales). Place its transaction workbook at `coffee-newsvendor/data/Coffee_Shop_Sales.xlsx`, or supply `--raw_xlsx /path/to/Coffee_Shop_Sales.xlsx`.

The transactions cover January to June 2023 at three New York City stores. For each coffee product sold in all three cup sizes, the sales of the three stores are summed, and the demand of a week is the average daily number of cups of each size. This gives 27 weekly observations for each of five products:

| Identifier | Product |
| --- | --- |
| `group_1` | Brazilian |
| `group_2` | Colombian |
| `group_3` | Ethiopian |
| `group_4` | Jamaican |
| `group_5` | Old Time Diner Blend |

## Experimental Setup

| Parameter | Value in Section 5.2 |
| --- | --- |
| Unit prices `p` | modal Brazilian prices, `[2.2, 3.0, 3.5]`, for all products |
| Order costs `c` | `[0.4, 1.9, 2.0]` |
| Holding costs `h` | `0.05 * p` |
| Backorder costs `b` | `1.5 * p` |
| Known target weeks | 5, drawn at random; the remaining 22 weeks are test weeks |
| Corruption of the other products | each week multiplied by 4 with probability 10%, 20%, 30% or 40% (one probability per product, in random order) |
| Reference distributions | the four corrupted histories and the known target weeks, K = 5 |
| Reference weights | 1/2 for the target, 1/8 for each other product |
| Grid of `delta` and `epsilon` | `{0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50}` |
| Grid of `nu` (RKHS) | `{0.05, 0.10, ..., 0.95}` |
| Barycenter samples | 128 |
| Repetitions | 10 |

Each product serves in turn as the target. Affine and Static use the radii `epsilon_k = (1 + delta) * W2(P_k, P_bary)`, where `P_bary` is the empirical distribution of 128 samples from the Gelbrich barycenter of the references; Barycenter is centered at the same `P_bary`. Pool, Mixture and Barycenter tune a single radius `epsilon`; Minkowski sets `epsilon_k = epsilon` for every reference. RKHS tunes `nu` and uses uniform reference weights. ERM has no hyperparameter.

The hyperparameter of each method is selected by leave-one-out cross-validation on the five known target weeks. In each fold, four weeks form the target reference distribution and the fifth is used for validation. Only hyperparameters for which the method solves to optimality in every fold are eligible. Validation costs that differ by less than a relative tolerance of `1e-6` count as ties, and the smallest tied hyperparameter is selected. The method is then re-solved with all five known weeks at the selected hyperparameter and evaluated on the 22 test weeks. In every repetition, the known weeks, the corruption, the barycenter samples and the RKHS randomization are redrawn. Every random quantity has its own seed derived from `--seed`, the target, the repetition and the fold, so the results do not depend on the number of workers. MOSEK and Gurobi use one thread per solve.

## Run

Follow the root installation instructions. Run from the `fusion-DRO` directory. The conic methods (MOSEK) and RKHS (Gurobi) are run as separate jobs that write to the same result folder:

```bash
python coffee-newsvendor/run_newsvendor.py \
  --outdir coffee-newsvendor/results/newsvendor \
  --workers 8 --mosek_license_file /path/to/mosek.lic

python coffee-newsvendor/run_newsvendor.py \
  --outdir coffee-newsvendor/results/newsvendor \
  --methods rkhs --workers 2 --gurobi_license_file /path/to/gurobi.lic
```

All settings of Section 5.2 are defaults. `--split_ids 0-4` restricts a job to some repetitions, for example to run the RKHS repetitions in two jobs.

## Results

Each instance (target and repetition) writes to `instances/<target>_split<r>/`:

- `data.npz`: the known and test weeks of the target, the corrupted histories and the weights.
- `cv_grid_<job>.csv`: validation cost, status and solve time of every method, hyperparameter and fold.
- `selected_<job>.csv`: the selected hyperparameter, the test cost and the time of the final solve of every method.

The table of Section 5.2 is built with

```bash
python coffee-newsvendor/make_table.py --results coffee-newsvendor/results/newsvendor
```

which writes `table.tex` and `table.csv` (means and standard errors) to the result folder. The reported time is the wall-clock time of the final solve at the selected hyperparameter, excluding cross-validation.

## Checks

From `fusion-DRO`, run `python -m unittest discover -s coffee-newsvendor -p "test_*.py"`. These checks cover the paper defaults, the newsvendor cost, the target split and the corruption; they need neither the data nor a solver license.

## References

* Xianyu Li, Huan Xu, Xiaolin Huang and Chao Shang. [Robust Out-of-Distribution Stochastic Optimization](https://arxiv.org/abs/2604.20147). *arXiv:2604.20147*. 2026.
