"""Checks of Section 5.3 that need neither the data nor a GPU:
python -m unittest discover -s credit-classification -p "test_*.py" (from fusion-DRO)."""
import unittest

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog

import credit_data as D
import domain as DM
import releases as R
from methods import tv_worst_case


class TestClassification(unittest.TestCase):
    def test_split(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"ID": np.arange(1, D.N_RECORDS + 1), "y": rng.integers(0, 2, D.N_RECORDS),
                           "PAY_0": rng.integers(-2, 9, D.N_RECORDS)})
        pool, test = D.split(df, 3)
        self.assertEqual((len(pool), len(test)), (D.N_POOL, D.N_TEST))
        self.assertEqual(len(np.union1d(pool, test)), D.N_RECORDS)

    def test_rounding_lands_in_its_cell(self):
        """random valid points of every histogram cell are one-hot and belong to that cell"""
        y, Rb, Fb, Sx, cells = DM.cell_cases(torch.device("cpu"))
        gen = torch.Generator().manual_seed(0)
        X = DM.random_valid((D.N_CELLS, 4), Rb[:, None], Fb[:, None], Sx[:, None], gen, "cpu", torch.float64)
        blocks = np.split(X[..., D.N_NUM:].reshape(-1, D.N_ONEHOT).numpy(), np.cumsum(D.LEVELS)[:-1], axis=1)
        self.assertTrue(all(np.all(b.sum(1) == 1) and np.all(b.max(1) == 1) for b in blocks))
        cats = np.stack([b.argmax(1) for b in blocks], 1)
        self.assertTrue(np.array_equal(D.cell(cats, y[:, None].expand(-1, 4).reshape(-1).numpy()),
                                       cells[:, None].expand(-1, 4).reshape(-1).numpy()))

    def test_tv_worst_case(self):
        """closed-form worst case over a total variation ball = linear program"""
        rng = np.random.default_rng(1)
        for r in (0.0, 0.05, 0.3, 1.0):
            v, p = rng.normal(size=12), rng.dirichlet(np.ones(12))
            t = lambda a: torch.as_tensor(np.atleast_1d(a), dtype=torch.float64)[None]
            q = tv_worst_case(t(v), t(p), t(r)[0])[0].numpy()
            # variables (q, u) with u >= |q - p|, sum q = 1, sum u <= 2 r
            n = len(v)
            A_ub = np.block([[np.eye(n), -np.eye(n)], [-np.eye(n), -np.eye(n)], [np.zeros(n), np.ones(n)]])
            lp = linprog(np.r_[-v, np.zeros(n)], A_ub=A_ub, b_ub=np.r_[p, -p, 2 * r],
                         A_eq=np.r_[np.ones(n), np.zeros(n)][None], b_eq=[1.0], bounds=(0, None))
            self.assertAlmostEqual(q @ v, -lp.fun, places=8)
            self.assertLessEqual(0.5 * np.abs(q - p).sum(), r + 1e-12)

    def test_privacy_calibration(self):
        sens = R.PHI_DIAM / 2000
        sigma = R.analytic_gaussian_sigma(0.5, 1e-6, sens)
        self.assertLessEqual(R.gaussian_delta(sigma, 0.5, sens), 1e-6)
        self.assertGreater(R.gaussian_delta(sigma * (1 - 1e-9), 0.5, sens), 1e-6)
        self.assertAlmostEqual(R.PHI_DIAM ** 2, 491.0)
        p = R.project_simplex(np.array([0.5, -0.2, 0.9, 0.1]))
        self.assertAlmostEqual(p.sum(), 1.0)
        self.assertTrue(np.all(p >= 0))


if __name__ == "__main__":
    unittest.main()
