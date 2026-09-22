from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from coffee_common import GROUP_COL


def _make_weights(k_total: int, target_weight: float) -> List[float]:
    if k_total <= 0:
        raise ValueError("k_total must be >= 1")
    if k_total == 1:
        return [1.0]
    if not (0.0 < target_weight < 1.0):
        raise ValueError("target_weight must be in (0,1) when k_total > 1")
    hist_w = (1.0 - float(target_weight)) / float(k_total - 1)
    return [float(hist_w)] * (k_total - 1) + [float(target_weight)]


def main() -> None:
    p = argparse.ArgumentParser(description="Build the five-reference schedule for one target coffee product.")
    p.add_argument("--processed_path", type=str, required=True, help="Prepared coffee_weekly_groups.xlsx path.")
    p.add_argument("--outdir", type=str, required=True, help="Output directory.")
    p.add_argument("--target_group", type=str, required=True, help="Held-out target group id, e.g. group_1.")
    p.add_argument("--k_start", type=int, choices=[5], default=5)
    p.add_argument("--k_max", type=int, choices=[5], default=5)
    p.add_argument("--target_weight", type=float, default=0.5, help="Weight assigned to the target-validation distribution.")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.processed_path, sheet_name="weekly_groups", engine="openpyxl")
    group_meta = pd.read_excel(args.processed_path, sheet_name="group_meta", engine="openpyxl")
    groups = sorted(df[GROUP_COL].astype(str).unique().tolist())
    target = str(args.target_group).strip()
    if target not in groups:
        raise ValueError(f"target_group={target} not found in processed data groups={groups}")

    historical = [g for g in groups if g != target]
    if len(groups) != 5:
        raise ValueError("The paper experiment requires five coffee products.")
    k_upper = 5
    schedule_df = pd.DataFrame([{
        "row_index": 0,
        "K_total": 5,
        "target_source": target,
        "selected_historical": json.dumps(historical),
        "source_values": json.dumps(historical + [target]),
        "source_roles": json.dumps(["historical"] * 4 + ["target_validation"]),
        "t_weights": json.dumps(_make_weights(5, float(args.target_weight))),
    }])

    meta = {
        "target_source": target,
        "all_groups": groups,
        "historical_candidates": historical,
        "k_start": int(args.k_start),
        "k_max_effective": int(k_upper),
        "target_weight": float(args.target_weight),
        "num_schedule_rows": int(schedule_df.shape[0]),
    }

    xlsx_path = outdir / "coffee_schedule.xlsx"
    csv_path = outdir / "coffee_schedule.csv"
    meta_path = outdir / "meta.json"

    schedule_df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        schedule_df.to_excel(writer, sheet_name="schedule", index=False)
        group_meta.to_excel(writer, sheet_name="group_meta", index=False)
        pd.DataFrame([meta]).to_excel(writer, sheet_name="meta", index=False)

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Target selected: {target}")
    print(f"K range: {int(args.k_start)}..{int(k_upper)}")
    print(f"Saved under {outdir}")


if __name__ == "__main__":
    main()
