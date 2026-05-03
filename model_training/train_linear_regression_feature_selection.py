import argparse
import glob
import logging
import os
import random
import sys
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mse
from darts.models import LinearRegressionModel

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import INPUT_COLS, PAST_COLS, TARGET_COLS, TIME_COL
from data_functionalities.detrend_data import (
    detrend_timeseries_linear,
    reset_detrend_storage,
    reverse_detrend_timeseries,
)
from data_functionalities.log_transform_data import (
    log_transform_timeseries,
    reset_log_transform_storage,
    reverse_log_transform_timeseries,
)


def _unique_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def load_blocks(data_dir: str, min_block_length: int, max_files: Optional[int] = None) -> List[pd.DataFrame]:
    block_frames: List[pd.DataFrame] = []
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if max_files is not None:
        all_files = all_files[:max_files]

    for filepath in all_files:
        df = pd.read_csv(filepath)
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        group_id = (df["session_index"] != df["session_index"].shift()).cumsum()

        file_blocks = 0
        for _, group_df in df.groupby(group_id):
            group_df = group_df.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL]).reset_index(drop=True)
            if len(group_df) < min_block_length:
                continue
            block_frames.append(group_df)
            file_blocks += 1

        print(f"Loaded {file_blocks} blocks from {filepath}")

    print(f"Total loaded blocks: {len(block_frames)}")
    return block_frames


def build_full_series_list(block_frames: List[pd.DataFrame], value_cols: Sequence[str]) -> List[TimeSeries]:
    series_list = []
    for block_df in block_frames:
        series_list.append(
            TimeSeries.from_dataframe(
                block_df[[TIME_COL, *value_cols]],
                time_col=TIME_COL,
                value_cols=list(value_cols),
            )
        )
    return series_list


def fit_scale_full_series_list(series_list: List[TimeSeries]) -> Tuple[List[TimeSeries], Scaler]:
    scaler = Scaler(global_fit=True)
    return scaler.fit_transform(series_list), scaler


def series_list_from_frames(frames: List[pd.DataFrame], columns: Sequence[str]) -> List[TimeSeries]:
    if not columns:
        return []

    subset_series = []
    for frame in frames:
        subset_series.append(TimeSeries.from_dataframe(frame[list(columns)]))
    return subset_series


def inverse_optional_transforms(
    prediction: TimeSeries,
    use_detrend: bool,
    use_log_transform: bool,
    ts_index: int,
) -> TimeSeries:
    output = prediction
    if use_detrend:
        output = reverse_detrend_timeseries([output], ts_indices=[ts_index])[0]
    if use_log_transform:
        output = reverse_log_transform_timeseries([output], ts_indices=[ts_index])[0]
    return output


def evaluate_feature_set(
    target_col: str,
    selected_future_cols: Sequence[str],
    selected_past_cols: Sequence[str],
    train_target_frames_scaled: List[pd.DataFrame],
    train_covariate_frames_scaled: List[pd.DataFrame],
    val_target_frames_scaled: List[pd.DataFrame],
    val_covariate_frames_scaled: List[pd.DataFrame],
    val_target_frames_raw: List[pd.DataFrame],
    target_scaler: Scaler,
    input_chunk_length: int,
    output_chunk_length: int,
    stride: int,
    use_detrend: bool,
    use_log_transform: bool,
    val_block_ids: Sequence[int],
) -> Tuple[float, float]:
    model_kwargs = {
        "lags": input_chunk_length,
        "output_chunk_length": output_chunk_length,
    }
    if selected_past_cols:
        model_kwargs["lags_past_covariates"] = input_chunk_length
    if selected_future_cols:
        model_kwargs["lags_future_covariates"] = (input_chunk_length, output_chunk_length)

    model = LinearRegressionModel(**model_kwargs)

    train_targets = series_list_from_frames(train_target_frames_scaled, [target_col])
    val_targets_scaled = series_list_from_frames(val_target_frames_scaled, [target_col])
    val_targets_raw = series_list_from_frames(val_target_frames_raw, [target_col])

    train_fit_kwargs = {"series": train_targets}
    if selected_future_cols:
        train_fit_kwargs["future_covariates"] = series_list_from_frames(train_covariate_frames_scaled, selected_future_cols)
    if selected_past_cols:
        train_fit_kwargs["past_covariates"] = series_list_from_frames(train_covariate_frames_scaled, selected_past_cols)

    model.fit(**train_fit_kwargs)

    val_future_covariates = None
    if selected_future_cols:
        val_future_covariates = series_list_from_frames(val_covariate_frames_scaled, selected_future_cols)

    val_past_covariates = None
    if selected_past_cols:
        val_past_covariates = series_list_from_frames(val_covariate_frames_scaled, selected_past_cols)

    mae_list = []
    mse_list = []

    for block_idx, (ts_target_scaled, ts_target_raw) in enumerate(zip(val_targets_scaled, val_targets_raw)):
        if len(ts_target_scaled) <= input_chunk_length + output_chunk_length:
            continue

        for forecast_start in range(input_chunk_length, len(ts_target_scaled) - output_chunk_length + 1, stride):
            input_start = forecast_start - input_chunk_length
            y_train = ts_target_scaled[input_start:forecast_start]
            y_true = ts_target_raw[forecast_start : forecast_start + output_chunk_length]

            predict_kwargs = {
                "n": output_chunk_length,
                "series": y_train,
                "verbose": False,
            }
            if selected_future_cols:
                predict_kwargs["future_covariates"] = val_future_covariates[block_idx]
            if selected_past_cols:
                predict_kwargs["past_covariates"] = val_past_covariates[block_idx]

            pred_scaled = model.predict(**predict_kwargs)
            pred = target_scaler.inverse_transform(pred_scaled)
            pred = inverse_optional_transforms(
                pred,
                use_detrend=use_detrend,
                use_log_transform=use_log_transform,
                ts_index=val_block_ids[block_idx],
            )

            mae_list.append(mae(y_true, pred))
            mse_list.append(mse(y_true, pred))

    if not mae_list:
        raise ValueError(
            f"No valid validation windows for target={target_col} and features={list(selected_future_cols) + list(selected_past_cols)}"
        )

    return sum(mae_list) / len(mae_list), sum(mse_list) / len(mse_list)


def greedy_feature_selection(
    target_col: str,
    train_target_frames_scaled: List[pd.DataFrame],
    train_covariate_frames_scaled: List[pd.DataFrame],
    val_target_frames_scaled: List[pd.DataFrame],
    val_covariate_frames_scaled: List[pd.DataFrame],
    val_target_frames_raw: List[pd.DataFrame],
    target_scaler: Scaler,
    input_chunk_length: int,
    output_chunk_length: int,
    stride: int,
    use_detrend: bool,
    use_log_transform: bool,
    val_block_ids: Sequence[int],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    future_candidates = _unique_preserve_order([col for col in INPUT_COLS if col != target_col])
    past_candidates = [col for col in _unique_preserve_order([col for col in PAST_COLS if col != target_col]) if col not in future_candidates]

    selected_future_cols: List[str] = []
    selected_past_cols: List[str] = []
    table_rows: List[Dict[str, object]] = []

    baseline_mae, baseline_mse = evaluate_feature_set(
        target_col=target_col,
        selected_future_cols=selected_future_cols,
        selected_past_cols=selected_past_cols,
        train_target_frames_scaled=train_target_frames_scaled,
        train_covariate_frames_scaled=train_covariate_frames_scaled,
        val_target_frames_scaled=val_target_frames_scaled,
        val_covariate_frames_scaled=val_covariate_frames_scaled,
        val_target_frames_raw=val_target_frames_raw,
        target_scaler=target_scaler,
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        stride=stride,
        use_detrend=use_detrend,
        use_log_transform=use_log_transform,
        val_block_ids=val_block_ids,
    )

    table_rows.append(
        {
            "step": 0,
            "added_variable": "baseline",
            "variable_type": "none",
            "decision": "baseline",
            "MAE": baseline_mae,
            "MSE": baseline_mse,
            "selected_future_covariates": "",
            "selected_past_covariates": "",
        }
    )

    current_best_mae = baseline_mae
    current_best_mse = baseline_mse
    step = 1

    candidate_stream = [("future", candidate) for candidate in future_candidates] + [
        ("past", candidate) for candidate in past_candidates
    ]

    for candidate_type, candidate in candidate_stream:
        if candidate_type == "future":
            trial_future = selected_future_cols + [candidate]
            trial_past = selected_past_cols
        else:
            trial_future = selected_future_cols
            trial_past = selected_past_cols + [candidate]

        trial_mae, trial_mse = evaluate_feature_set(
            target_col=target_col,
            selected_future_cols=trial_future,
            selected_past_cols=trial_past,
            train_target_frames_scaled=train_target_frames_scaled,
            train_covariate_frames_scaled=train_covariate_frames_scaled,
            val_target_frames_scaled=val_target_frames_scaled,
            val_covariate_frames_scaled=val_covariate_frames_scaled,
            val_target_frames_raw=val_target_frames_raw,
            target_scaler=target_scaler,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            stride=stride,
            use_detrend=use_detrend,
            use_log_transform=use_log_transform,
            val_block_ids=val_block_ids,
        )

        improved = trial_mae < current_best_mae
        if improved:
            if candidate_type == "future":
                selected_future_cols.append(candidate)
            else:
                selected_past_cols.append(candidate)
            current_best_mae = trial_mae
            current_best_mse = trial_mse

        print(
            f"  Checked {candidate} ({candidate_type}) - MAE: {trial_mae:.4f} "
            f"{'[added]' if improved else '[skipped]'}"
        )

        table_rows.append(
            {
                "step": step,
                "added_variable": candidate,
                "variable_type": candidate_type,
                "decision": "added" if improved else "skipped",
                "MAE": trial_mae,
                "MSE": trial_mse,
                "selected_future_covariates": ", ".join(selected_future_cols),
                "selected_past_covariates": ", ".join(selected_past_cols),
            }
        )
        step += 1

    table = pd.DataFrame(table_rows)
    return table, selected_future_cols, selected_past_cols


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy feature selection for LinearRegressionModel using INPUT_COLS and PAST_COLS."
    )
    parser.add_argument("--data-dir", default="data_preprocessed", help="Directory with preprocessed CSV files.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit on the number of CSV files.")
    parser.add_argument("--min-block-length", type=int, default=720, help="Minimum block length to keep.")
    parser.add_argument("--input-chunk-length", type=int, default=60)
    parser.add_argument("--output-chunk-length", type=int, default=30)
    parser.add_argument("--stride", type=int, default=6, help="Stride for validation window evaluation.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle blocks before split.")
    parser.add_argument("--use-detrend", action="store_true", help="Apply linear detrending before training.")
    parser.add_argument("--use-log-transform", action="store_true", help="Apply log transform before training.")
    args = parser.parse_args()

    print(
        f"use_detrend={args.use_detrend}, use_log_transform={args.use_log_transform}, "
        f"input_chunk_length={args.input_chunk_length}, output_chunk_length={args.output_chunk_length}"
    )

    block_frames = load_blocks(args.data_dir, args.min_block_length, args.max_files)
    if not block_frames:
        raise RuntimeError("No valid blocks found.")

    if args.shuffle:
        random.seed(42)
        random.shuffle(block_frames)

    full_value_cols = _unique_preserve_order([*TARGET_COLS, *INPUT_COLS, *PAST_COLS])
    full_series_list = build_full_series_list(block_frames, full_value_cols)

    if args.use_log_transform:
        reset_log_transform_storage()
        print("\nApplying log transform to all selected columns...")
        full_series_list = log_transform_timeseries(full_series_list)
        print("Log transform completed.")

    if args.use_detrend:
        reset_detrend_storage()
        print("\nDetrending all selected columns...")
        full_series_list = detrend_timeseries_linear(full_series_list)
        print("Detrending completed.")

    raw_frames = [series.to_dataframe() for series in full_series_list]

    output_dir = os.path.join("outputs", "linear_regression_feature_tables")
    os.makedirs(output_dir, exist_ok=True)

    results: Dict[str, Dict[str, object]] = {}

    for target_col in TARGET_COLS:
        print("\n" + "=" * 70)
        print(f"Selecting features for target: {target_col}")
        print("=" * 70)

        n_total = len(full_series_list)
        split_idx_1 = int(n_total * 0.7)
        split_idx_2 = int(n_total * 0.85)

        train_raw_frames = raw_frames[:split_idx_1]
        val_raw_frames = raw_frames[split_idx_1:split_idx_2]
        test_raw_frames = raw_frames[split_idx_2:]

        covariate_cols = [col for col in _unique_preserve_order([*INPUT_COLS, *PAST_COLS]) if col != target_col]

        train_target_frames_raw = [frame[[target_col]] for frame in train_raw_frames]
        val_target_frames_raw = [frame[[target_col]] for frame in val_raw_frames]
        test_target_frames_raw = [frame[[target_col]] for frame in test_raw_frames]

        train_covariate_frames_raw = [frame[covariate_cols] for frame in train_raw_frames]
        val_covariate_frames_raw = [frame[covariate_cols] for frame in val_raw_frames]
        test_covariate_frames_raw = [frame[covariate_cols] for frame in test_raw_frames]

        train_target_series = series_list_from_frames(train_target_frames_raw, [target_col])
        val_target_series = series_list_from_frames(val_target_frames_raw, [target_col])
        test_target_series = series_list_from_frames(test_target_frames_raw, [target_col])

        train_covariate_series = series_list_from_frames(train_covariate_frames_raw, covariate_cols)
        val_covariate_series = series_list_from_frames(val_covariate_frames_raw, covariate_cols)
        test_covariate_series = series_list_from_frames(test_covariate_frames_raw, covariate_cols)

        train_target_series_scaled, target_scaler = fit_scale_full_series_list(train_target_series)
        val_target_series_scaled = target_scaler.transform(val_target_series)
        test_target_series_scaled = target_scaler.transform(test_target_series)

        train_covariate_series_scaled, covariate_scaler = fit_scale_full_series_list(train_covariate_series)
        val_covariate_series_scaled = covariate_scaler.transform(val_covariate_series)
        test_covariate_series_scaled = covariate_scaler.transform(test_covariate_series)

        train_target_frames_scaled = [series.to_dataframe() for series in train_target_series_scaled]
        val_target_frames_scaled = [series.to_dataframe() for series in val_target_series_scaled]
        test_target_frames_scaled = [series.to_dataframe() for series in test_target_series_scaled]

        train_covariate_frames_scaled = [series.to_dataframe() for series in train_covariate_series_scaled]
        val_covariate_frames_scaled = [series.to_dataframe() for series in val_covariate_series_scaled]
        test_covariate_frames_scaled = [series.to_dataframe() for series in test_covariate_series_scaled]

        val_block_ids = list(range(split_idx_1, split_idx_2))

        print(
            f"Train blocks: {len(train_target_frames_scaled)}, Val blocks: {len(val_target_frames_scaled)}, "
            f"Test blocks: {len(test_target_frames_scaled)}"
        )

        table, selected_future_cols, selected_past_cols = greedy_feature_selection(
            target_col=target_col,
            train_target_frames_scaled=train_target_frames_scaled,
            train_covariate_frames_scaled=train_covariate_frames_scaled,
            val_target_frames_scaled=val_target_frames_scaled,
            val_covariate_frames_scaled=val_covariate_frames_scaled,
            val_target_frames_raw=val_target_frames_raw,
            target_scaler=target_scaler,
            input_chunk_length=args.input_chunk_length,
            output_chunk_length=args.output_chunk_length,
            stride=args.stride,
            use_detrend=args.use_detrend,
            use_log_transform=args.use_log_transform,
            val_block_ids=val_block_ids,
        )

        table_path = os.path.join(output_dir, f"{target_col}_feature_selection.csv")
        table.to_csv(table_path, index=False)

        results[target_col] = {
            "table": table,
            "future": selected_future_cols,
            "past": selected_past_cols,
            "table_path": table_path,
        }

        print(table.to_string(index=False))
        print(
            f"Selected future covariates for {target_col}: {', '.join(selected_future_cols) if selected_future_cols else '<none>'}"
        )
        print(
            f"Selected past covariates for {target_col}: {', '.join(selected_past_cols) if selected_past_cols else '<none>'}"
        )

    summary_rows = []
    for target_col, info in results.items():
        summary_rows.append(
            {
                "target": target_col,
                "future_covariates": ", ".join(info["future"]),
                "past_covariates": ", ".join(info["past"]),
                "table_path": info["table_path"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "selection_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nFinal tables summary:")
    for _, row in summary_df.iterrows():
        print(f"\nTarget: {row['target']}")
        print(f"  Future covariates: {row['future_covariates'] if row['future_covariates'] else '<none>'}")
        print(f"  Past covariates: {row['past_covariates'] if row['past_covariates'] else '<none>'}")
        print(f"  Table saved to: {row['table_path']}")


if __name__ == "__main__":
    main()