"""Checks of the Section 5.2 setup that need neither data nor solver licenses.

    python -m unittest discover -s coffee-newsvendor -p "test_*.py"     (from fusion-DRO)
"""
from __future__ import annotations

import unittest

import numpy as np

import run_newsvendor as rn
from coffee_data import branch_matrices, cost_parameters, newsvendor_cost_matrix, parse_float_list


class NewsvendorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.W = {f"group_{k}": rng.uniform(5, 50, size=(27, 3)) for k in range(1, 6)}
        self.cp = cost_parameters(np.array([2.2, 3.0, 3.5]))

    def test_paper_defaults(self):
        a = rn.build_parser().parse_args([])
        self.assertEqual(parse_float_list(a.delta_grid), [0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50])
        self.assertEqual(parse_float_list(a.epsilon_grid), parse_float_list(a.delta_grid))
        np.testing.assert_allclose(parse_float_list(a.nu_grid), np.arange(1, 20) * 0.05)
        self.assertEqual((a.n_splits, a.n_known, a.target_weight, a.barycenter_num_samples), (10, 5, 0.5, 128))
        np.testing.assert_allclose(self.cp["C"], [0.4, 1.9, 2.0])
        np.testing.assert_allclose(self.cp["H"], 0.05 * self.cp["P"])
        np.testing.assert_allclose(self.cp["B"], 1.5 * self.cp["P"])

    def test_cost_equals_maximum_of_affine_pieces(self):
        A_X, A_Xi = branch_matrices(self.cp)
        self.assertEqual(A_X.shape[0], 2 ** 3)
        rng = np.random.default_rng(1)
        X, w = rng.uniform(0, 60, size=(50, 3)), rng.uniform(0, 60, size=3)
        np.testing.assert_allclose(newsvendor_cost_matrix(X, w, self.cp), np.max(A_X @ w + X @ A_Xi.T, axis=1))

    def test_instance(self):
        known, test, others, hist, t = rn.make_instance(self.W, "group_2", 3, 1234, 5, 0.5)
        self.assertEqual((len(known), len(test)), (5, 22))
        self.assertEqual(sorted(np.concatenate([known, test])), list(range(27)))
        self.assertNotIn("group_2", others)
        np.testing.assert_allclose(t, [0.125] * 4 + [0.5])
        for h, X in zip(others, hist):
            ratio = X / self.W[h]
            self.assertTrue(np.all(np.isclose(ratio, 1.0) | np.isclose(ratio, rn.BULK_FACTOR)))
        again = rn.make_instance(self.W, "group_2", 3, 1234, 5, 0.5)
        np.testing.assert_array_equal(known, again[0])
        for X, Y in zip(hist, again[3]):
            np.testing.assert_array_equal(X, Y)

    def test_bulk_probabilities(self):
        rng = np.random.default_rng(2)
        X = np.ones((200_000, 3))
        for level, prob in enumerate(rn.BULK_PROBABILITIES):
            share = np.mean(rn.corrupt(X, level, rng)[:, 0] == rn.BULK_FACTOR)
            self.assertAlmostEqual(share, prob, delta=0.005)


if __name__ == "__main__":
    unittest.main()
