"""Fit and persist Darts scalers for an existing checkpoint folder.

This is useful for older `darts_logs/<model_name>/` runs that don't yet contain
`scalers.pkl`, but you want MPC to use identical scaling as training.

This script fits global scalers from `data_preprocessed/*.csv` using the same
block segmentation as `model_training/train_models.py` (session_index changes).

Example:
  python -m mpc.fit_scalers \
    --model-name NeuralForecast_Nhits_I60_O30 \
    --input-cols clinker_1_feedrate separator_speed \
    --target-cols return first_chamber_filling second_chamber_filling gran1_blain
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

from config import TIME_COL


def load_blocks(
    *,
    filepath: str | Path,
    target_cols: list[str],
    input_cols: list[str],
) -> tuple[list[TimeSeries], list[TimeSeries] | None]:
    df = pd.read_csv(filepath)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    group_id = (df["session_index"] != df["session_index"].shift()).cumsum()

    targets: list[TimeSeries] = []
    covs: list[TimeSeries] = []

    for _, group_df in df.groupby(group_id):
        group_df = group_df.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL])
        if len(group_df) < 720:
            continue

        targets.append(
            TimeSeries.from_dataframe(group_df, time_col=TIME_COL, value_cols=target_cols)
        )

        if input_cols:
            covs.append(
                TimeSeries.from_dataframe(group_df, time_col=TIME_COL, value_cols=input_cols)
            )

    return targets, (covs if input_cols else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-dir", default="darts_logs")
    parser.add_argument("--data-dir", default="data_preprocessed")

    parser.add_argument("--target-cols", nargs="+", required=True)
    parser.add_argument("--input-cols", nargs="+", default=[])

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    targets_all: list[TimeSeries] = []
    covs_all: list[TimeSeries] = []

    for fp in files:
        targets, covs = load_blocks(filepath=fp, target_cols=args.target_cols, input_cols=args.input_cols)
        targets_all.extend(targets)
        if covs is not None:
            covs_all.extend(covs)

    if not targets_all:
        raise RuntimeError("No valid blocks found (all were <720 samples?)")

    target_scaler = Scaler(global_fit=True)
    cov_scaler = Scaler(global_fit=True) if args.input_cols else None

    target_scaler.fit(targets_all)
    if cov_scaler is not None:
        cov_scaler.fit(covs_all)

    out_path = Path(args.model_dir) / args.model_name / "scalers.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "target_scaler": target_scaler,
        "covariates_scaler": cov_scaler,
        "meta": {
            "target_cols": list(args.target_cols),
            "input_cols": list(args.input_cols),
        },
    }

    with out_path.open("wb") as f:
        pickle.dump(bundle, f)

    print(f"Saved scalers to: {out_path}")


if __name__ == "__main__":
    main()
