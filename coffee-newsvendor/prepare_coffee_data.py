from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from coffee_common import (
    BASE_COL,
    COUNT_COL,
    DEMAND_COLS,
    GROUP_COL,
    RAW_COFFEE_XLSX,
    WEEK_COL,
    build_weekly_group_data,
    load_cost_params,
    make_long_weekly_frame,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare weekly demand and observed unit prices from the Maven coffee transactions.")
    p.add_argument("--raw_xlsx", type=str, default=str(RAW_COFFEE_XLSX), help="Raw coffee sales workbook.")
    p.add_argument("--outdir", type=str, default="", help="Output directory. Defaults to coffee-newsvendor/data/prepared.")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    outdir = Path(args.outdir) if str(args.outdir).strip() else script_dir / "data" / "prepared"
    outdir.mkdir(parents=True, exist_ok=True)

    group_frames, price_df = build_weekly_group_data(args.raw_xlsx)

    long_df = make_long_weekly_frame(group_frames)
    processed_xlsx = outdir / "coffee_weekly_groups.xlsx"
    long_csv = outdir / "coffee_weekly_groups_long.csv"
    price_csv = outdir / "unit_price_by_coffee_size.csv"
    meta_json = outdir / "meta.json"

    with pd.ExcelWriter(processed_xlsx, engine="openpyxl") as writer:
        long_df.to_excel(writer, sheet_name="weekly_groups", index=False)
        price_df.to_excel(writer, sheet_name="unit_prices", index=False)
        pd.DataFrame(
            [
                {
                    GROUP_COL: frame[GROUP_COL].iloc[0],
                    BASE_COL: frame[BASE_COL].iloc[0],
                    "n_weeks": int(frame.shape[0]),
                }
                for frame in group_frames
            ]
        ).to_excel(writer, sheet_name="group_meta", index=False)

    long_df.to_csv(long_csv, index=False)
    price_df.to_csv(price_csv, index=False)

    cost_params = load_cost_params(price_csv)
    meta = {
        "raw_xlsx": str(Path(args.raw_xlsx).resolve()),
        "processed_xlsx": str(processed_xlsx.resolve()),
        "long_csv": str(long_csv.resolve()),
        "price_csv": str(price_csv.resolve()),
        "group_count": int(len(group_frames)),
        "weeks_per_group": {str(frame[GROUP_COL].iloc[0]): int(frame.shape[0]) for frame in group_frames},
        "cost_params_assumption": {
            "price_row_used": 0,
            "P": [float(v) for v in cost_params["P"].tolist()],
            "C": [float(v) for v in cost_params["C"].tolist()],
            "H": [float(v) for v in cost_params["H"].tolist()],
            "B": [float(v) for v in cost_params["B"].tolist()],
        },
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Processed data written to {outdir}")
    print(f"Workbook: {processed_xlsx}")
    print(f"Price CSV: {price_csv}")


if __name__ == "__main__":
    main()
