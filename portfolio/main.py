from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

if __package__ in {None, ''}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio.cli import build_parser
else:
    from .cli import build_parser


def _timestamped_output_root(cfg):
    parent = Path(cfg.out_dir).expanduser()
    if not parent.is_absolute():
        parent = Path(__file__).resolve().parent / parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = f'res_w2_{timestamp}_syn'
    candidate = parent / name
    suffix = 1
    while candidate.exists():
        candidate = parent / f'{name}_{suffix}'
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, 'mosek_license_path', None):
        os.environ['MOSEKLM_LICENSE_FILE'] = str(Path(args.mosek_license_path).expanduser())

    if args.mode == 'replot':
        import pandas as pd
        from portfolio.plotting.plot_scaling import plot_scaling

        results = Path(args.results_dir).expanduser()
        destination = Path(args.out_dir).expanduser() if args.out_dir else results
        plot_scaling(pd.read_csv(results / 'scaling.csv'), destination, timestamp_source=results)
        print(f'Figures saved in {destination}')
        return

    from portfolio.config import build_config
    from portfolio.io_utils import save_dataframe, save_json
    from portfolio.runners.run_scaling import run_scaling

    cfg = build_config(args)
    output = _timestamped_output_root(cfg)
    save_json(cfg.to_dict(), output / 'config.json')
    df = run_scaling(cfg, out_path=output / 'scaling.csv')
    save_dataframe(df, output / 'scaling.csv')
    if cfg.plot:
        from portfolio.plotting.plot_scaling import plot_scaling
        plot_scaling(df, output)
    print(f'Done. Results saved in {output}')


if __name__ == '__main__':
    main()
