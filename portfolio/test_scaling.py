"""Regression checks: python -m unittest portfolio.test_scaling (from fusion-DRO)."""
from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import cvxpy as cp

from .cli import build_parser
from .config import build_config, normalized_source_weights
from .data_gen import generate_populations
from .runners.run_scaling import _job_cfg
from .solver_retry import solve_with_scaled_retry


class ScalingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = build_config(build_parser().parse_args(['scaling']))

    def test_paper_defaults(self):
        c = self.cfg
        self.assertEqual((c.sigma_sys, c.sigma_id), (0.02, 0.015))
        self.assertEqual((c.mu_high, c.mu_low, c.mu_high_growth, c.mu_low_shrink),
                         (0.015, -0.005, 0.001, 0.001))
        self.assertEqual(c.K_grid, list(range(2, 21)))
        self.assertEqual(c.N_list, [5, 10, *range(20, 101, 10)])
        self.assertEqual((c.num_runs, c.seed_base, c.cov_direction_seed), (10, 0, 12345))
        self.assertTrue(c.cov_direction_per_run)
        self.assertEqual(normalized_source_weights(c, 3), [1 / 3] * 3)

    def test_only_paper_commands_are_exposed(self):
        parser = build_parser()
        subcommands = next(action for action in parser._actions
                           if isinstance(action, argparse._SubParsersAction))
        self.assertEqual(set(subcommands.choices), {'scaling', 'replot'})

    def test_unused_commands_and_options_are_rejected(self):
        for args in (['oos'], ['sensitivity'], ['scaling', '--run-rkhs', 'True'],
                     ['scaling', '--data', 'real'], ['scaling', '--wass-order', '1']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    build_parser().parse_args(args)

    def test_covariance_resampling_and_prefixes(self):
        full_cfg = _job_cfg(self.cfg, K_total=20, N_per_dist=10)
        small_cfg = _job_cfg(self.cfg, K_total=3, N_per_dist=10)
        first = generate_populations(full_cfg, 0)
        repeat = generate_populations(full_cfg, 0)
        next_run = generate_populations(full_cfg, 1)
        prefix = generate_populations(small_cfg, 0)
        for key in ('means', 'covariances', 'samples'):
            np.testing.assert_array_equal(first[key], repeat[key])
            np.testing.assert_array_equal(first[key][:3], prefix[key])
        np.testing.assert_array_equal(first['means'], next_run['means'])
        self.assertFalse(np.array_equal(first['covariances'], next_run['covariances']))
        self.assertEqual(len({tuple(g) for g in first['groups']}), 20)
        for a, b in zip(first['groups'], first['groups'][1:]):
            self.assertEqual(len(set(a) & set(b)), 2)

    def test_center_sample_count_tracks_each_job(self):
        self.assertEqual(_job_cfg(self.cfg, K_total=3, N_per_dist=50).barycenter_num_samples, 50)

    def test_scaled_retry_preserves_weights_and_original_units(self):
        seen = []
        def solver(**kw):
            seen.append(kw)
            if len(seen) == 1:
                raise cp.error.SolverError('UNKNOWN')
            return {'status': 'optimal', 'objective': 50.0, 'tau': 2.0,
                    'weights': np.array([0.4, 0.6])}
        train = {'source_samples': np.array([[[0.01, 0.02]]])}
        with patch('portfolio.solver_retry.time.perf_counter', side_effect=[0.0, 3.0, 8.0]):
            result = solve_with_scaled_retry(solver, train_data=train, cfg=self.cfg,
                                             wass_order=2, eps_cost=np.array([0.01]))
        np.testing.assert_allclose(seen[1]['train_data']['source_samples'], [[[1.0, 2.0]]])
        np.testing.assert_allclose(seen[1]['eps_cost'], [100.0])
        np.testing.assert_array_equal(result['weights'], [0.4, 0.6])
        self.assertEqual(result['objective'], 0.5)
        self.assertEqual(result['tau'], 0.02)
        self.assertEqual(result['reported_solve_time_sec'], 5.0)

    def test_only_joint_figures_with_parent_timestamp(self):
        from .plotting import plot_scaling as plotting
        data = pd.DataFrame({'method': ['msdro', 'affine']})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plotting, '_configure_plot_style'), \
                 patch.object(plotting, '_plot_joint_log_time_curves') as time_plot, \
                 patch.object(plotting, '_plot_joint_objective_curves') as objective_plot:
                plotting.plot_scaling(data, directory,
                                      timestamp_source='res_w2_20260729_094721_syn')
            self.assertEqual(time_plot.call_args.args[1].name, 'time_vs_log_joint_20260729_094721.pdf')
            self.assertEqual(objective_plot.call_args.args[1].name, 'objective_vs_joint_20260729_094721.pdf')
            self.assertEqual(time_plot.call_args.args[0]['method'].tolist(), ['exact', 'affine'])
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
