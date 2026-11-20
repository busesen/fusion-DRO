# Neural Network Training for Classification: Section 5.3

This experiment trains a neural network credit-default classifier without access to any records. The data holder releases differentially private moments of the numerical features and a differentially private histogram of the categorical features, and each release defines an ambiguity set. Affine, Static, Minkowski, Moment and TV are trained with the subgradient method of Section 4 and compared with Oracle, which trains on the hidden records themselves.

## Code

The optimization code is plain Python and PyTorch and runs on one GPU (or, slowly, on a CPU). The Slurm scripts in `cluster/` only submit many such runs.

| File | Content |
| --- | --- |
| `credit_data.py` | data source, download and checksum, preprocessing, histogram cells, moment map, random splits |
| `releases.py` | noisy moments and noisy histogram with their radii, nonemptiness certificate of the fusion ambiguity set |
| `domain.py` | the support set Xi: projection, rounding to valid one-hot encodings, random starting points |
| `methods.py` | the network and the objectives of all methods, including the inner maximizations |
| `sgd.py` | the subgradient method for the robust methods, stochastic gradient descent for Oracle, test cross-entropy |
| `run_classification.py` | entry point: one N, a list of simulations and a list of methods |
| `make_table.py` | Table 3 from the results |
| `cluster/` | `job.sbatch` (one job) and `submit_all.sh` (all jobs of the paper) |

## Data

The experiment uses the [Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) of the UCI Machine Learning Repository (CC BY 4.0). It records whether each of 30,000 credit card clients in Taiwan defaults in the following month. The zip file is not included. It is downloaded from `https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip` into `credit-classification/data/` on first use, and the code checks its SHA-256 (`56c885f8...727602`). On a cluster whose compute nodes have no internet access, download it once on the login node:

```bash
python credit-classification/credit_data.py
```

No records are dropped or imputed, as the file has no missing values. Preprocessing (`credit_data.py`) uses only fixed constants:

- **14 numerical features**: the credit limit, the age, six monthly bill amounts and six monthly payment amounts. Each is clipped to a fixed range, monetary amounts are log-transformed (signed for bill amounts, which can be negative), and the result is mapped onto `[-1, 1]`. Clipping changes one value of the dataset.
- **9 categorical features**: sex, education, marital status and six monthly repayment statuses, one-hot encoded into 45 network inputs. Codes that the UCI documentation does not define are merged: education 0, 5 and 6 into "other", marital status 0 into "other", and repayment statuses of three or more months late into one level.
- **Histogram cells** (432): the label, sex, education, marital status, the latest repayment status (current, one month late, at least two months late) and the number of late months among the five earlier months (zero, one or two, three to five).
- **Moments** (134): the distinct first and second moments of the numerical features and the label. The data holder computes them from the numerical features rounded to the midpoints of fixed bins (`EDGES` in `credit_data.py`).

## Experimental Setup

| Parameter | Value in Section 5.3 |
| --- | --- |
| Hidden records N | 2,000, 6,000, 10,000, 14,000, 18,000 |
| Simulations | 50 |
| Split per simulation | stratified (label x latest repayment status) random split into a pool of 18,000 and 12,000 test records; the N hidden records are the first N of a random order of the pool, so they are nested in N |
| Moment release | analytic Gaussian mechanism, (0.5, 1e-6)-differential privacy, radius `sigma * sqrt(chi2_{134}(0.95))` |
| Histogram release | Laplace mechanism on the proportions, Euclidean projection onto the simplex, (0.5, 0)-differential privacy, radius = 95% quantile of the total variation distance between the noisy and the true proportions (Monte Carlo, 4 million draws) |
| Nonemptiness | certified in every simulation by a distribution on finitely many points of Xi (the first 2,000 pool records combined with both labels) |
| Network | 59 -> 256 -> 128 -> 64 -> 1, SiLU activations, output logit capped as `8 tanh(g / 8)` |
| Weights | `t_1 = t_2 = 1/2` |
| Decision rules | Affine: `a_k^T s(xi) + <A_k, chi chi^T>` with `chi = (x, y)` (59 inputs and the label) and `s = (chi, cell indicators)`; Static: `A_k = 0` |
| Updates of the robust methods | 75,000, constant step size 0.05 (network, moment multipliers, cell coefficients) and 0.0125 (other decision rule coefficients) |
| Inner maximizations | projected gradient ascent with 8 starts and 32 steps, then rounding to a valid point; starts: the previous maximizer, 3 perturbations of it, 4 random points |
| Oracle | minibatch SGD on the hidden records (batch 128, step size 0.05), stopped at the checkpoint (2,500, then every 5,000 updates up to 75,000) with the lowest test cross-entropy |

The robust methods use no records. Every evaluation of their objective solves one maximization over Xi for the moment set (one per histogram cell for Affine and Static, whose rules have cell coefficients) and one per histogram cell for the histogram set. All methods of a simulation start from the same network initialization.

## Run

Install the requirements (PyTorch with GPU support as appropriate for the machine):

```bash
python -m pip install -r credit-classification/requirements.txt
```

From the `fusion-DRO` directory, one call trains the listed methods for one N on the listed simulations, all in parallel on one GPU:

```bash
python credit-classification/run_classification.py --n 6000 --sims 0-9 --methods affine,static
```

The first call for an N computes and certifies the releases of each simulation and caches them in `results/releases/`. `--releases_only` does only this. On a GPU, the inner maximizations are replayed from a CUDA graph with TF32 matrix products; `--no_cuda_graph` turns the graph off. On NVIDIA GH200 GPUs, ten simulations of Affine take about 2.1 hours and ten of Static about 2.0 hours.

The full experiment of the paper consists of 80 such jobs (16 per N). On a Slurm cluster, `cluster/submit_all.sh` submits all of them, with one releases job per N that the training jobs wait for:

```bash
bash credit-classification/cluster/submit_all.sh
```

Add the account, partition and environment settings of the cluster to `cluster/job.sbatch`.

## Results

Each call writes `results/n<N>/<method>_sims<a>-<b>.csv` with the test cross-entropy of every simulation, and, for Oracle, the selected update. For example, the call in [Run](#run) writes `results/n6000/affine_sims0-9.csv` and `results/n6000/static_sims0-9.csv`. Table 3 (mean and standard error of `100 (CE / CE_Oracle - 1)` over the simulations) is built with

```bash
python credit-classification/make_table.py --results credit-classification/results
```

which writes `table.csv` and `table.tex` to the result folder.

The data, the splits and the releases are deterministic and do not depend on the hardware. Training on a GPU is not bitwise reproducible, and the random starting points of the inner maximizations are drawn from one generator per job, so a rerun reproduces the table up to simulation noise rather than digit for digit.

## Checks

From `fusion-DRO`, run `python -m unittest discover -s credit-classification -p "test_*.py"`. These checks cover the split, the rounding into histogram cells, the closed-form worst case over the histogram set and the privacy calibration; they need neither the data nor a GPU.
