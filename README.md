# fusion-DRO

This repository contains the Python implementations used to generate the numerical experiments in the paper **Efficient Data Fusion in Distributionally Robust Optimization**.

| Folder | Paper section | Experiment |
| --- | --- | --- |
| [portfolio](portfolio/README.md) | 5.1 | Solve times and robust objective values for synthetic portfolio optimization |
| [coffee-newsvendor](coffee-newsvendor/README.md) | 5.2 | Out-of-sample multi-item newsvendor costs on coffee sales data |

Both experiments use 2-Wasserstein ambiguity sets with squared Euclidean transport cost. Affine is solved as an SDP; the other conic formulations are solved with MOSEK.

## Installation

Use Python 3.10 or newer. From this directory:

```bash
python -m pip install -r requirements.txt
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic
```

Gurobi is used only for the optional coffee RKHS benchmark of Li et al. (2026).
Its dependencies are included in the regular requirements. Enable it with
`--run_rkhs True` and `--gurobi_license_file /path/to/gurobi.lic`.

Follow the experiment READMEs for the paper's parameters, run commands, and outputs.
