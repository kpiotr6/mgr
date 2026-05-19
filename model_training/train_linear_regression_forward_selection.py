"""
Train Darts LinearRegressionModel with forward selection over PAST_COLS and INPUT_COLS.
Uses backward (past covariates) and forward (future covariates) windows.
"""

import argparse
import glob
import logging
import os
import random
import sys
import warnings
from typing import List, Optional, Tuple

import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import rmse
from darts.models import LinearRegressionModel

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import INPUT_COLS, PAST_COLS, TARGET_COLS, TIME_COL
from data_functionalities.detrend_data import (
    detrend_timeseries_linear,
    reset_detrend_storage,
)
from data_functionalities.log_transform_data import (
    log_transform_timeseries,
    reset_log_transform_storage,
)


def load_data(filepath: str) -> Tuple[List[TimeSeries], List[Optional[TimeSeries]], List[Optional[TimeSeries]]]:
    df = pd.read_csv(filepath)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    value_cols = list(dict.fromkeys(TARGET_COLS + INPUT_COLS + PAST_COLS))
    for col in value_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    group_id = (df["session_index"] != df["session_index"].shift()).cumsum()

    targets = []
    covariates = []
    past_covariates = []

    for _, group_df in df.groupby(group_id):
        group_df = group_df.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL])
        if len(group_df) < 720:
            continue

        target_ts = TimeSeries.from_dataframe(
            group_df,
            time_col=TIME_COL,
            value_cols=TARGET_COLS,
        )
        targets.append(target_ts)

        cov_ts = None
        if len(INPUT_COLS) > 0:
            cov_ts = TimeSeries.from_dataframe(
                group_df,
                time_col=TIME_COL,
                value_cols=INPUT_COLS,
            )
        covariates.append(cov_ts)

        past_cov_ts = None
        if len(PAST_COLS) > 0:
            past_cov_ts = TimeSeries.from_dataframe(
                group_df,
                time_col=TIME_COL,
                value_cols=PAST_COLS,
            )
        past_covariates.append(past_cov_ts)

    return targets, covariates, past_covariates


def select_covariates(
    covariates_list: List[Optional[TimeSeries]],
    selected_cols: List[str],
) -> List[Optional[TimeSeries]]:
    if not selected_cols:
        return [None] * len(covariates_list)
    return [ts[selected_cols] if ts is not None else None for ts in covariates_list]


def evaluate_model(
    train_targets: List[TimeSeries],
    val_targets: List[TimeSeries],
    train_covariates: List[Optional[TimeSeries]],
    val_covariates: List[Optional[TimeSeries]],
    train_past_covariates: List[Optional[TimeSeries]],
    val_past_covariates: List[Optional[TimeSeries]],
    input_chunk_length: int,
    output_chunk_length: int,
) -> float:
    model_kwargs = {
        "lags": input_chunk_length,
        "output_chunk_length": output_chunk_length,
    }
    if any(ts is not None for ts in train_past_covariates):
        model_kwargs["lags_past_covariates"] = input_chunk_length
    if any(ts is not None for ts in train_covariates):
        model_kwargs["lags_future_covariates"] = (input_chunk_length, output_chunk_length)

    model = LinearRegressionModel(**model_kwargs)

    fit_kwargs = {
        "series": train_targets,
        "val_series": val_targets,
    }
    if any(ts is not None for ts in train_covariates):
        fit_kwargs["future_covariates"] = train_covariates
        fit_kwargs["val_future_covariates"] = val_covariates
    if any(ts is not None for ts in train_past_covariates):
        fit_kwargs["past_covariates"] = train_past_covariates
        fit_kwargs["val_past_covariates"] = val_past_covariates

    model.fit(**fit_kwargs)

    selected_indices = [
        idx
        for idx, t in enumerate(val_targets)
        if len(t) > input_chunk_length + output_chunk_length
    ]
    if not selected_indices:
        return float("inf")

    val_series_for_pred = [
        val_targets[idx][:-output_chunk_length]
        for idx in selected_indices
    ]
    val_true = [
        val_targets[idx][-output_chunk_length:]
        for idx in selected_indices
    ]

    predict_kwargs = {
        "n": output_chunk_length,
        "series": val_series_for_pred,
        "verbose": False,
    }
    if any(ts is not None for ts in val_covariates):
        predict_kwargs["future_covariates"] = [val_covariates[idx] for idx in selected_indices]
    if any(ts is not None for ts in val_past_covariates):
        predict_kwargs["past_covariates"] = [val_past_covariates[idx] for idx in selected_indices]

    preds = model.predict(**predict_kwargs)
    score = rmse(val_true, preds)
    if isinstance(score, list):
        return float(sum(score) / len(score)) if score else float("inf")
    return float(score)


def forward_selection(
    train_targets: List[TimeSeries],
    val_targets: List[TimeSeries],
    train_covariates_full: List[Optional[TimeSeries]],
    val_covariates_full: List[Optional[TimeSeries]],
    train_past_covariates_full: List[Optional[TimeSeries]],
    val_past_covariates_full: List[Optional[TimeSeries]],
    input_chunk_length: int,
    output_chunk_length: int,
    max_features: Optional[int],
    min_improvement: float,
) -> Tuple[List[str], List[str], List[dict]]:
    selected_past = []
    selected_input = []
    best_score = float("inf")
    steps = []

    candidates = [("past", col) for col in PAST_COLS] + [("input", col) for col in INPUT_COLS]

    for candidate_type, candidate in candidates:
        if max_features is not None and (len(selected_past) + len(selected_input)) >= max_features:
            break

        if candidate_type == "past":
            candidate_past = selected_past + [candidate]
            candidate_input = selected_input
        else:
            candidate_past = selected_past
            candidate_input = selected_input + [candidate]

        train_past_cov = select_covariates(train_past_covariates_full, candidate_past)
        val_past_cov = select_covariates(val_past_covariates_full, candidate_past)
        train_cov = select_covariates(train_covariates_full, candidate_input)
        val_cov = select_covariates(val_covariates_full, candidate_input)

        score = evaluate_model(
            train_targets,
            val_targets,
            train_cov,
            val_cov,
            train_past_cov,
            val_past_cov,
            input_chunk_length,
            output_chunk_length,
        )

        improvement = best_score - score
        if improvement >= min_improvement:
            best_score = score
            if candidate_type == "past":
                selected_past.append(candidate)
            else:
                selected_input.append(candidate)

            steps.append(
                {
                    "step": len(steps) + 1,
                    "added_type": candidate_type,
                    "added_feature": candidate,
                    "rmse": best_score,
                    "selected_past": ",".join(selected_past),
                    "selected_input": ",".join(selected_input),
                }
            )

    return selected_past, selected_input, steps


def main():
    parser = argparse.ArgumentParser(
        description="Forward selection for LinearRegressionModel with past/future covariates."
    )
    parser.add_argument("--input-chunk-length", type=int, default=60)
    parser.add_argument("--output-chunk-length", type=int, default=30)
    parser.add_argument(
        "--input-chunk-lengths",
        type=str,
        default=None,
        help="Comma-separated input lengths (e.g., 60,120). Overrides --input-chunk-length.",
    )
    parser.add_argument(
        "--output-chunk-lengths",
        type=str,
        default=None,
        help="Comma-separated output lengths (e.g., 15,30). Overrides --output-chunk-length.",
    )
    parser.add_argument("--data-dir", type=str, default="data_preprocessed")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--min-improvement", type=float, default=1e-4)
    parser.add_argument("--use-detrend", action="store_true")
    parser.add_argument("--use-log-transform", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    random.seed(args.seed)

    all_files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if args.max_files is not None:
        all_files = all_files[: args.max_files]

    targets_list = []
    covariates_list = []
    past_covariates_list = []

    for filepath in all_files:
        t, c, p = load_data(filepath)
        targets_list.extend(t)
        covariates_list.extend(c)
        past_covariates_list.extend(p)
        print(f"Loaded {len(t)} series blocks from {filepath}")

    print(f"Total loaded series blocks: {len(targets_list)}")

    if args.use_log_transform:
        reset_log_transform_storage()
        print("Applying log transform...")
        targets_list = log_transform_timeseries(targets_list)
        if len(INPUT_COLS) > 0:
            covariates_list = log_transform_timeseries(covariates_list)
        if len(PAST_COLS) > 0:
            past_covariates_list = log_transform_timeseries(past_covariates_list)
        print("Log transform completed.")

    if args.use_detrend:
        reset_detrend_storage()
        print("Detrending data...")
        targets_list = detrend_timeseries_linear(targets_list)
        if len(INPUT_COLS) > 0:
            covariates_list = detrend_timeseries_linear(covariates_list)
        if len(PAST_COLS) > 0:
            past_covariates_list = detrend_timeseries_linear(past_covariates_list)
        print("Detrending completed.")

    if args.shuffle:
        combined = list(zip(targets_list, covariates_list, past_covariates_list))
        random.shuffle(combined)
        targets_list, covariates_list, past_covariates_list = zip(*combined)
        targets_list = list(targets_list)
        covariates_list = list(covariates_list)
        past_covariates_list = list(past_covariates_list)

    n_total = len(targets_list)
    split_idx_1 = int(n_total * 0.7)
    split_idx_2 = int(n_total * 0.85)

    train_targets = targets_list[:split_idx_1]
    train_covariates = covariates_list[:split_idx_1]
    train_past_covariates = past_covariates_list[:split_idx_1]

    val_targets = targets_list[split_idx_1:split_idx_2]
    val_covariates = covariates_list[split_idx_1:split_idx_2]
    val_past_covariates = past_covariates_list[split_idx_1:split_idx_2]

    print(
        f"Train blocks: {len(train_targets)}, Val blocks: {len(val_targets)}, "
        f"Test blocks: {len(targets_list[split_idx_2:])}"
    )

    target_scaler = Scaler(global_fit=True)
    covariates_scaler = Scaler(global_fit=True) if len(INPUT_COLS) > 0 else None
    past_covariates_scaler = Scaler(global_fit=True) if len(PAST_COLS) > 0 else None

    train_targets_scaled = target_scaler.fit_transform(train_targets)
    val_targets_scaled = target_scaler.transform(val_targets)

    if len(INPUT_COLS) > 0:
        train_covariates_scaled = covariates_scaler.fit_transform(train_covariates)
        val_covariates_scaled = covariates_scaler.transform(val_covariates)
    else:
        train_covariates_scaled = [None] * len(train_targets)
        val_covariates_scaled = [None] * len(val_targets)

    if len(PAST_COLS) > 0:
        train_past_covariates_scaled = past_covariates_scaler.fit_transform(train_past_covariates)
        val_past_covariates_scaled = past_covariates_scaler.transform(val_past_covariates)
    else:
        train_past_covariates_scaled = [None] * len(train_targets)
        val_past_covariates_scaled = [None] * len(val_targets)

    os.makedirs(args.output_dir, exist_ok=True)

    input_lengths = (
        [int(x) for x in args.input_chunk_lengths.split(",") if x.strip()]
        if args.input_chunk_lengths
        else [args.input_chunk_length]
    )
    output_lengths = (
        [int(x) for x in args.output_chunk_lengths.split(",") if x.strip()]
        if args.output_chunk_lengths
        else [args.output_chunk_length]
    )

    all_summaries = []

    for target_col in TARGET_COLS:
        print(f"\n=== Target: {target_col} ===")
        train_targets_target = [ts[target_col] for ts in train_targets_scaled]
        val_targets_target = [ts[target_col] for ts in val_targets_scaled]

        for input_len in input_lengths:
            for output_len in output_lengths:
                print(f"\nRunning forward selection for I={input_len}, O={output_len}")
                selected_past, selected_input, steps = forward_selection(
                    train_targets_target,
                    val_targets_target,
                    train_covariates_scaled,
                    val_covariates_scaled,
                    train_past_covariates_scaled,
                    val_past_covariates_scaled,
                    input_len,
                    output_len,
                    args.max_features,
                    args.min_improvement,
                )

                steps_df = pd.DataFrame(steps)
                steps_path = os.path.join(
                    args.output_dir,
                    f"linear_regression_forward_selection_steps_{target_col}_I{input_len}_O{output_len}.csv",
                )
                steps_df.to_csv(steps_path, index=False)

                summary_row = {
                    "target": target_col,
                    "input_chunk_length": input_len,
                    "output_chunk_length": output_len,
                    "selected_past": ",".join(selected_past),
                    "selected_input": ",".join(selected_input),
                    "final_rmse": steps[-1]["rmse"] if steps else None,
                }
                all_summaries.append(summary_row)

                print("Forward selection completed.")
                print(f"Selected past covariates: {selected_past}")
                print(f"Selected input covariates: {selected_input}")
                print(f"Steps saved to {steps_path}")

    summary_path = os.path.join(
        args.output_dir,
        "linear_regression_forward_selection_summary.csv",
    )
    pd.DataFrame(all_summaries).to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
