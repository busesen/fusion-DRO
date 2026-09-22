"""Integration checks: python coffee-newsvendor/test_pipeline.py."""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import random
import unittest
from unittest.mock import patch

import numpy as np

import build_coffee_schedule
import empirical_solver_coffee
import rkhs
import run_coffee_pipeline
import run_coffee_solver_grid


class ParserCaptured(Exception):
    pass


def capture_parser(main):
    parsers = []
    def intercept(parser, *args, **kwargs):
        parsers.append(parser)
        raise ParserCaptured()
    with patch.object(argparse.ArgumentParser, 'parse_args', intercept):
        try:
            main()
        except ParserCaptured:
            pass
    return parsers[0]


class CoffeeTests(unittest.TestCase):
    def test_rkhs_is_opt_in_at_every_entry_point(self):
        for module in (run_coffee_pipeline, run_coffee_solver_grid, empirical_solver_coffee):
            with self.subTest(module=module.__name__):
                parser = capture_parser(module.main)
                self.assertEqual(parser.get_default('run_rkhs'), 'False')
                self.assertEqual(parser.get_default('rkhs_source_weights'), 'uniform')

    def test_five_reference_schedule_weights(self):
        self.assertEqual(build_coffee_schedule._make_weights(5, 0.5), [0.125] * 4 + [0.5])
        parser = capture_parser(run_coffee_pipeline.main)
        for args in (['--k_start', '4'], ['--wass_p', '1'], ['--affine_static_center', 'mixture']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(args)

    def test_rkhs_weights_are_uniform(self):
        mode, weights = empirical_solver_coffee._resolve_rkhs_source_weights([0.125] * 4 + [0.5], 'uniform')
        self.assertEqual(mode, 'uniform')
        np.testing.assert_array_equal(weights, np.full(5, 0.2))


class RkhsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rkhs.check_rkhs_dependencies()

    def setUp(self):
        rng = np.random.RandomState(123)
        self.samples = [rng.uniform(1, 5, (4, 3)) + shift for shift in (0, 0.5, 1)]
        self.costs = dict(H=np.array([0.11, 0.15, 0.175]), B=np.array([1.1, 1.5, 1.75]),
                          C=np.array([0.4, 1.9, 2.0]), P=np.array([2.2, 3.0, 3.5]), dim=3)

    def solve(self, **overrides):
        args = dict(samples_list=self.samples, validation_samples=self.samples[0],
                    cost_params=self.costs, outdir='', verbose=False, seed=123,
                    rkhs_nu=0.3, rkhs_source_weights=[1/3] * 3,
                    rkhs_source_weight_mode='uniform', rkhs_n_J=8,
                    rkhs_is_herding=False, rkhs_T=3, rkhs_time_limit=30)
        args.update(overrides)
        return rkhs.solve_rkhs(**args)

    def test_gaussian_reference_gram_matches_direct_average(self):
        pooled = np.vstack(self.samples)
        distances = np.linalg.norm(pooled[:, None, :] - pooled[None, :, :], axis=2)
        gamma = 1 / (2 * np.median(distances) ** 2)
        self.assertAlmostEqual(rkhs.median_heuristic(pooled, pooled)[1], gamma)
        expected = np.array([
            [np.exp(-gamma * ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)).mean()
             for y in self.samples] for x in self.samples
        ])
        np.testing.assert_allclose(rkhs.getKernel(self.samples, gamma), expected)

    def test_radius_with_and_without_boundary_support(self):
        gram, alpha = np.eye(2), np.full(2, 0.5)
        for boundary in (np.array([0, 1]), np.array([], dtype=int)):
            with patch.object(rkhs, 'solvedual', return_value=(alpha, [0, 1], boundary, [])):
                _, radius = rkhs.SMC(gram, 2, [], 0.5, source_weights=[0.5, 0.5])
            self.assertAlmostEqual(radius, np.sqrt(0.5))

    def test_herding_uses_empirical_points(self):
        samples = np.array([[0.0], [1.0], [2.0]])
        actual = rkhs.kernel_herding('Gaussian', [1.0, 1.0], samples, T=5)
        self.assertEqual(actual.shape, (5, 1))
        self.assertEqual(actual[0, 0], 1.0)
        self.assertTrue(all(row in samples for row in actual))

    def test_latin_hypercube_keeps_existing_unseeded_constructor(self):
        with patch.object(rkhs, 'SMC', return_value=[np.full(3, 1/3), 0.1]):
            with patch.object(rkhs.qmc, 'LatinHypercube', side_effect=RuntimeError('stop before solving')) as sampler:
                status, objective, order, _ = self.solve()
        sampler.assert_called_once_with(3)
        self.assertEqual(status, 'error: stop before solving')
        self.assertIsNone(objective)
        self.assertIsNone(order)

    def test_adapter_restores_global_rng_states(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        def draw(*args, **kwargs):
            return np.random.uniform(size=3), random.random(), [0, 0, 0]
        with patch.object(rkhs, 'M_SMC', side_effect=draw):
            first = self.solve()
            second = self.solve()
        np.testing.assert_array_equal(first[2], second[2])
        self.assertEqual(first[1], second[1])
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(np.random.get_state()[2:], numpy_state[2:])

    def test_adapter_preserves_existing_status_contract(self):
        for order, expected in ((np.ones(3), 'optimal'), (None, 'failed')):
            with patch.object(rkhs, 'M_SMC', return_value=(order, 7.0, [0, 0, 0])):
                status, objective, _, extra = self.solve()
            self.assertEqual(status, expected)
            self.assertEqual(objective, 7.0)
            self.assertEqual(extra['source_weights'], [1/3] * 3)
        with patch.object(rkhs, 'M_SMC', side_effect=RuntimeError('original solver error')):
            status, objective, order, _ = self.solve()
        self.assertEqual(status, 'error: original solver error')
        self.assertIsNone(objective)
        self.assertIsNone(order)

    @unittest.skipUnless(os.environ.get('FUSION_DRO_TEST_GUROBI') == '1', 'requires a Gurobi license')
    def test_original_routines_numerical_regression(self):
        sampler = rkhs.qmc.LatinHypercube
        # Control support only in this test to compare against the original routines.
        # The production sampler remains unseeded, exactly as in the existing code.
        with patch.object(rkhs.qmc, 'LatinHypercube', side_effect=lambda dim: sampler(dim, seed=123)):
            for herding, expected in ((False, -19.7068129476), (True, -18.7132107060)):
                with self.subTest(herding=herding):
                    status, objective, order, _ = self.solve(rkhs_is_herding=herding)
                    self.assertEqual(status, 'optimal')
                    self.assertAlmostEqual(objective, expected, places=5)
                    self.assertTrue(np.all(order >= -1e-6))
                    repeated = self.solve(rkhs_is_herding=herding, validation_samples=np.full((2, 3), 1e8))
                    self.assertAlmostEqual(repeated[1], objective, places=6)
                    np.testing.assert_allclose(repeated[2], order)


if __name__ == '__main__':
    unittest.main()
