import os
import glob
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import warnings
import json

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae
from darts.models import LinearRegressionModel

# Allow running this file directly (e.g. `python model_training/train_models.py`)
# by ensuring the project root (where `config.py` lives) is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Import configurations from your config file
from config import INPUT_COLS, PAST_COLS, TARGET_COLS, TIME_COL

warnings.filterwarnings("ignore")

def load_all_series(filepath: str, min_length: int):
    """Loads a single CSV and returns a list of dataframes grouped by session_index."""
    df = pd.read_csv(filepath)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    # Identify distinct series blocks based on session_index
    group_id = (df['session_index'] != df['session_index'].shift()).cumsum()

    series_dfs = []
    for _, group_df in df.groupby(group_id):
        group_df = group_df.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL])
        if len(group_df) > min_length:
            series_dfs.append(group_df)

    return series_dfs

def extract_timeseries(dfs: list, cols: list) -> list:
    """Converts a list of dataframes into a list of Darts TimeSeries for specific columns."""
    if not cols:
        return None
    return [TimeSeries.from_dataframe(df, time_col=TIME_COL, value_cols=cols) for df in dfs]

def slice_timeseries(ts_list: list, cols: list) -> list:
    """Helper to slice specific columns from a list of TimeSeries."""
    if not cols or ts_list is None:
        return None
    return [ts[cols] for ts in ts_list]

def run_forward_selection():
    # 1. Define windows and parameters
    input_chunk_lengths = [6]
    output_chunk_lengths = [2]
    min_series_length = max(input_chunk_lengths) + max(output_chunk_lengths)

    # Dictionary to store the final optimal features for output generation
    final_config = {}

    # 2. Load all raw data
    data_dir = "data_preprocessed"
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    all_dfs = []
    for filepath in all_files:
        all_dfs.extend(load_all_series(filepath, min_series_length))

    print(f"Total loaded series blocks: {len(all_dfs)}")

    # 3. Train/Val Split (70% train, 15% val, ignoring test for feature selection)
    n_total = len(all_dfs)
    split_idx_1 = int(n_total * 0.7)
    split_idx_2 = int(n_total * 0.85)

    train_dfs = all_dfs[:split_idx_1]
    val_dfs = all_dfs[split_idx_2:n_total]

    # 4. Global Scaler for Covariates (Independent of target scaler)
    all_feature_cols = list(set(TARGET_COLS + PAST_COLS + INPUT_COLS))

    train_ts_all = extract_timeseries(train_dfs, all_feature_cols)
    val_ts_all = extract_timeseries(val_dfs, all_feature_cols)

    cov_scaler = Scaler(global_fit=True)
    train_scaled = cov_scaler.fit_transform(train_ts_all)
    val_scaled = cov_scaler.transform(val_ts_all)

    # 5. Pipeline execution
    for target in TARGET_COLS:
        print(f"\n{'='*50}\nStarting Forward Selection for Target: {target}\n{'='*50}")

        # Tracking the best setup for the current target across all windows tested
        target_best_overall_mae = float('inf')
        target_best_config = {"past": [], "input": []}

        # Dedicated scaler strictly for the current target
        target_scaler = Scaler(global_fit=True)
        train_target_raw = extract_timeseries(train_dfs, [target])
        val_target_raw = extract_timeseries(val_dfs, [target])

        train_target_ts = target_scaler.fit_transform(train_target_raw)
        val_target_ts = target_scaler.transform(val_target_raw)

        # Define candidate features
        past_candidates = list(PAST_COLS) + [t for t in TARGET_COLS if t != target]
        future_candidates = list(INPUT_COLS)

        for in_chunk in input_chunk_lengths:
            for out_chunk in output_chunk_lengths:
                print(f"\n--- Testing Window: In={in_chunk}, Out={out_chunk} ---")

                selected_past = []
                selected_future = []
                best_mae = float('inf')

                remaining_past = list(past_candidates)
                remaining_future = list(future_candidates)

                while remaining_past or remaining_future:
                    step_best_mae = float('inf')
                    step_best_feature = None
                    step_best_type = None

                    candidates_to_test = [('past', f) for f in remaining_past] + [('future', f) for f in remaining_future]

                    for f_type, feature in candidates_to_test:
                        test_past_cols = selected_past + ([feature] if f_type == 'past' else [])
                        test_fut_cols = selected_future + ([feature] if f_type == 'future' else [])

                        # Slice datasets for the current feature combination
                        train_p = slice_timeseries(train_scaled, test_past_cols) if test_past_cols else None
                        train_f = slice_timeseries(train_scaled, test_fut_cols) if test_fut_cols else None

                        val_p = slice_timeseries(val_scaled, test_past_cols) if test_past_cols else None
                        val_f = slice_timeseries(val_scaled, test_fut_cols) if test_fut_cols else None

                        # Initialize Linear Regression Model
                        model = LinearRegressionModel(
                            lags=in_chunk,
                            lags_past_covariates=in_chunk if test_past_cols else None,
                            lags_future_covariates=(in_chunk, out_chunk) if test_fut_cols else None,
                            output_chunk_length=out_chunk
                        )

                        # Fit model
                        fit_kwargs = {"series": train_target_ts}
                        if train_p: fit_kwargs["past_covariates"] = train_p
                        if train_f: fit_kwargs["future_covariates"] = train_f

                        try:
                            model.fit(**fit_kwargs)
                        except Exception as e:
                            continue

                        # Evaluate on validation set
                        current_maes = []
                        for i in range(len(val_target_ts)):
                            ts_val_scaled = val_target_ts[i]
                            if len(ts_val_scaled) < in_chunk + out_chunk:
                                continue

                            # Iterate over the whole timeseries with stride=1
                            for j in range(len(ts_val_scaled) - in_chunk - out_chunk + 1):
                                y_train = ts_val_scaled[j : j + in_chunk]
                                y_true = val_target_raw[i][j + in_chunk : j + in_chunk + out_chunk]

                                pred_kwargs = {
                                    "n": out_chunk,
                                    "series": y_train,
                                    "verbose": False
                                }
                                if val_p: pred_kwargs["past_covariates"] = val_p[i][j : j + in_chunk]
                                if val_f: pred_kwargs["future_covariates"] = val_f[i][j : j + in_chunk + out_chunk]

                                pred_scaled = model.predict(**pred_kwargs)

                                # Inverse transform prediction using the isolated target scaler
                                pred = target_scaler.inverse_transform(pred_scaled)

                                current_maes.append(mae(y_true, pred))

                        if not current_maes:
                            continue

                        avg_mae = sum(current_maes) / len(current_maes)

                        if avg_mae < step_best_mae:
                            step_best_mae = avg_mae
                            step_best_feature = feature
                            step_best_type = f_type

                    # Process outcome of the current forward selection step
                    if step_best_mae < best_mae:
                        best_mae = step_best_mae
                        print(f"  [+] Added '{step_best_feature}' ({step_best_type}) | New Best MAE: {best_mae:.4f}")

                        if step_best_type == 'past':
                            selected_past.append(step_best_feature)
                            remaining_past.remove(step_best_feature)
                        else:
                            selected_future.append(step_best_feature)
                            remaining_future.remove(step_best_feature)
                    else:
                        print(f"  [*] No further improvement. Stopping forward selection.")
                        break

                # Update best config for the target if this window combination is better
                if best_mae < target_best_overall_mae:
                    target_best_overall_mae = best_mae
                    target_best_config["past"] = sorted(selected_past)
                    target_best_config["input"] = sorted(selected_future)

        # Save the best configuration for formatting later
        final_config[target] = target_best_config

    # =====================================================================
    # 6. Generate Requested Code Output
    # =====================================================================
    print("\n\n" + "="*50)
    print("FINAL EXTRACTED CONFIGURATION")
    print("="*50 + "\n")

    # Safely print dictionary using json.dumps for perfect syntax and spacing
    print(f"PER_TARGET_CONFIG = {json.dumps(final_config, indent=2)}\n")

    def print_list_block(name, lst):
        print(f"{name} = [")
        for col in lst:
            print(f'    "{col}",')
        print("]\n")

    print_list_block("INPUT_COLS", INPUT_COLS)
    print_list_block("TARGET_COLS", TARGET_COLS)
    print_list_block("PAST_COLS", PAST_COLS)

if __name__ == "__main__":
    run_forward_selection()