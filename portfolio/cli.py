from __future__ import annotations

import argparse


def _parse_bool_value(raw):
    value = str(raw).strip().lower()
    if value in {'true', '1', 'yes', 'on'}:
        return True
    if value in {'false', '0', 'no', 'off'}:
        return False
    raise argparse.ArgumentTypeError('Expected True or False.')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Section 5.1: synthetic W2 portfolio scaling.')
    commands = parser.add_subparsers(dest='mode', required=True)
    scaling = commands.add_parser(
        'scaling',
        help='Run the paper experiment with a fixed full-family center in the K sweep.',
    )
    scaling.add_argument('--data', choices=['synthetic'], default='synthetic')
    scaling.add_argument('--wass-order', type=int, choices=[2], default=2)
    scaling.add_argument('--d-assets', type=int, default=10)
    scaling.add_argument('--K-grid', default=','.join(map(str, range(2, 21))))
    scaling.add_argument('--N-list', default='5,10,20,30,40,50,60,70,80,90,100')
    scaling.add_argument('--K-for-N-sweep', type=int, default=3)
    scaling.add_argument('--N-for-K-sweep', type=int, default=10)
    scaling.add_argument('--num-runs', type=int, default=10)
    scaling.add_argument('--seed-base', type=int, default=0)
    scaling.add_argument('--sigma-sys', type=float, default=0.02)
    scaling.add_argument('--sigma-id', type=float, default=0.015)
    scaling.add_argument('--cov-pattern', choices=['heterogeneous'], default='heterogeneous')
    scaling.add_argument('--cov-direction-seed', type=int, default=12345)
    scaling.add_argument('--cov-direction-per-run', action=argparse.BooleanOptionalAction, default=True,
                         help='Redraw covariance directions for each repetition (paper default).')
    scaling.add_argument('--mean-pattern', choices=['combinatorial_overlap_blocks'],
                         default='combinatorial_overlap_blocks')
    scaling.add_argument('--block-size', type=int, default=4)
    scaling.add_argument('--stride', type=int, default=2, help='Number of replaced assets per block.')
    scaling.add_argument('--mu-high', type=float, default=0.015)
    scaling.add_argument('--mu-low', type=float, default=-0.005)
    scaling.add_argument('--mu-high-growth', type=float, default=0.001)
    scaling.add_argument('--mu-low-shrink', type=float, default=0.001)
    scaling.add_argument('--rho', type=float, default=10.0)
    scaling.add_argument('--eta', type=float, default=0.2)
    scaling.add_argument('--fixed-delta', type=float, default=0.01)
    scaling.add_argument('--approx-mode', choices=['both', 'affine', 'static'], default='both')
    for method in ('exact', 'mixture', 'minkowski'):
        scaling.add_argument(f'--run-{method}', nargs='?', const=True, default=True, type=_parse_bool_value)
    scaling.add_argument('--max-tuples', default='200000', help='Exact tuple cap, or none.')
    scaling.add_argument('--max-exact-constraints', default='35000', help='Exact constraint cap, or none.')
    scaling.add_argument('--exact-time-limit-sec', type=float, default=1800.0)
    scaling.add_argument('--solver', default='MOSEK')
    scaling.add_argument('--mosek-license-path', default=None)
    scaling.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False)
    scaling.add_argument('--plot', action=argparse.BooleanOptionalAction, default=True)
    scaling.add_argument('--out-dir', default='results',
                         help='Result parent directory; relative paths are resolved against portfolio/.')

    replot = commands.add_parser('replot', help='Regenerate Figures 1-2 from scaling.csv, without solving.')
    replot.add_argument('--results-dir', required=True)
    replot.add_argument('--out-dir', default=None, help='Optional separate destination for regenerated figures.')
    return parser
