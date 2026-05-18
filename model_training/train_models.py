import pandas as pd
import argparse
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, rmse, mape
import os
import sys
import glob
import random
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config import INPUT_COLS, PAST_COLS, TARGET_COLS, TIME_COL

from data_functionalities.detrend_data import (
    detrend_timeseries_linear,
    reverse_detrend_timeseries,
    reset_detrend_storage
)
from data_functionalities.log_transform_data import (
    log_transform_timeseries,
    reverse_log_transform_timeseries,
    reset_log_transform_storage,
)
from model_definitions import (
    get_models,
    NAIVE_MODELS,
    MODELS_WITH_FUTURE_COVARIATES,
    MODELS_FUTURE_COVARIATES_ONLY,
    CHECKPOINT_MODELS
)


def load_data(filepath: str):
    df = pd.read_csv(filepath)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    # A new time series starts when 'session_index' changes
    group_id = (df['session_index'] != df['session_index'].shift()).cumsum()

    targets = []
    covariates = []
    past_covariates = []

    has_input_covariates = len(INPUT_COLS) > 0
    has_past_covariates = len(PAST_COLS) > 0

    for _, group_df in df.groupby(group_id):
        group_df = group_df.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL])

        if len(group_df) < 720:
            continue

        # Create targets TimeSeries
        target_ts = TimeSeries.from_dataframe(
            group_df,
            time_col=TIME_COL,
            value_cols=TARGET_COLS,
        )
        targets.append(target_ts)

        # Create future covariates TimeSeries
        cov_ts = None
        if has_input_covariates:
            cov_ts = TimeSeries.from_dataframe(
                group_df,
                time_col=TIME_COL,
                value_cols=INPUT_COLS,
            )
        covariates.append(cov_ts)

        # Create past covariates TimeSeries
        past_cov_ts = None
        if has_past_covariates:
            past_cov_ts = TimeSeries.from_dataframe(
                group_df,
                time_col=TIME_COL,
                value_cols=PAST_COLS,
            )
        past_covariates.append(past_cov_ts)

    return targets, covariates, past_covariates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train forecasting models with optional preprocessing transforms.")
    parser.add_argument(
        "--use-detrend",
        action="store_true",
        help="Apply linear detrending before training and reverse it after prediction.",
    )
    parser.add_argument(
        "--use-log-transform",
        action="store_true",
        help="Apply log transform before training and reverse it after prediction.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the data blocks before train/val/test split.",
    )
    args = parser.parse_args()

    use_detrend = args.use_detrend
    use_log_transform = args.use_log_transform
    shuffle_data = args.shuffle
    has_input_covariates = len(INPUT_COLS) > 0
    has_past_covariates = len(PAST_COLS) > 0

    print(f"use_detrend={use_detrend}, use_log_transform={use_log_transform}")

    targets_list = []
    covariates_list = []
    past_covariates_list = []

    # Number of files to use for dataset creation (None to use all files)
    MAX_FILES_TO_LOAD = None

    data_dir = "data_preprocessed"
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))

    all_files = sorted(all_files)
    if MAX_FILES_TO_LOAD is not None:
        all_files = all_files[:MAX_FILES_TO_LOAD]

    for filepath in all_files:
        t, c, p = load_data(filepath)
        targets_list.extend(t)
        covariates_list.extend(c)
        past_covariates_list.extend(p)
        print(f"Loaded {len(t)} series blocks from {filepath}")

    print(f"Total loaded series blocks: {len(targets_list)}")

    # Keep original targets for final metric computation/export on original scale
    original_targets_list = list(targets_list)
    series_indices = list(range(len(targets_list)))

    if use_log_transform:
        # Ensure no stale transform state from previous runs
        reset_log_transform_storage()
        print("\nApplying log transform...")
        targets_list = log_transform_timeseries(targets_list)
        if has_input_covariates:
            covariates_list = log_transform_timeseries(covariates_list)
        if has_past_covariates:
            past_covariates_list = log_transform_timeseries(past_covariates_list)
        print("Log transform completed.")

    if use_detrend:
        # Ensure no stale transform state from previous runs
        reset_detrend_storage()
        print("\nDetrending data...")
        targets_list = detrend_timeseries_linear(targets_list)
        if has_input_covariates:
            covariates_list = detrend_timeseries_linear(covariates_list)
        if has_past_covariates:
            past_covariates_list = detrend_timeseries_linear(past_covariates_list)
        print("Data detrending completed.")

    # Optionally shuffle the blocks to randomize train/test datasets
    if shuffle_data:
        random.seed(42)
        combined = list(zip(targets_list, covariates_list, past_covariates_list, original_targets_list, series_indices))
        random.shuffle(combined)
        targets_list, covariates_list, past_covariates_list, original_targets_list, series_indices = zip(*combined)

    # Needs to be a list again
    targets_list = list(targets_list)
    covariates_list = list(covariates_list)
    past_covariates_list = list(past_covariates_list)
    original_targets_list = list(original_targets_list)
    series_indices = list(series_indices)

    # Split into train, val, test (70% / 15% / 15%)
    n_total = len(targets_list)
    split_idx_1 = int(n_total * 0.7)
    split_idx_2 = int(n_total * 0.85)

    train_targets = targets_list[:split_idx_1]
    train_covariates = covariates_list[:split_idx_1]
    train_past_covariates = past_covariates_list[:split_idx_1]

    val_targets = targets_list[split_idx_1:split_idx_2]
    val_covariates = covariates_list[split_idx_1:split_idx_2]
    val_past_covariates = past_covariates_list[split_idx_1:split_idx_2]

    test_targets = targets_list[split_idx_2:]
    test_covariates = covariates_list[split_idx_2:]
    test_past_covariates = past_covariates_list[split_idx_2:]
    test_targets_raw = original_targets_list[split_idx_2:]
    test_series_indices = series_indices[split_idx_2:]

    print(f"Train blocks: {len(train_targets)}, Val blocks: {len(val_targets)}, Test blocks: {len(test_targets)}")

    # Scale the data
    target_scaler = Scaler(global_fit=True)
    covariates_scaler = Scaler(global_fit=True) if has_input_covariates else None
    past_covariates_scaler = Scaler(global_fit=True) if has_past_covariates else None

    train_targets_scaled = target_scaler.fit_transform(train_targets)
    train_covariates_scaled = covariates_scaler.fit_transform(train_covariates) if has_input_covariates else [None] * len(train_targets)
    train_past_covariates_scaled = past_covariates_scaler.fit_transform(train_past_covariates) if has_past_covariates else [None] * len(train_targets)

    val_targets_scaled = target_scaler.transform(val_targets)
    val_covariates_scaled = covariates_scaler.transform(val_covariates) if has_input_covariates else [None] * len(val_targets)
    val_past_covariates_scaled = past_covariates_scaler.transform(val_past_covariates) if has_past_covariates else [None] * len(val_targets)

    test_targets_scaled = target_scaler.transform(test_targets)
    test_covariates_scaled = covariates_scaler.transform(test_covariates) if has_input_covariates else [None] * len(test_targets)
    test_past_covariates_scaled = past_covariates_scaler.transform(test_past_covariates) if has_past_covariates else [None] * len(test_targets)

    INPUT_CHUNK_LENGTHS = [60, 120]
    OUTPUT_CHUNK_LENGTHS = [15, 30, 60]
    all_results = []
    best_models_dict = {}
    true_windows_for_chunk = []

    for INPUT_CHUNK_LENGTH in INPUT_CHUNK_LENGTHS:
        for OUTPUT_CHUNK_LENGTH in OUTPUT_CHUNK_LENGTHS:
            print(f"\n" + "="*50)
            print(f"Training for INPUT_CHUNK_LENGTH={INPUT_CHUNK_LENGTH}, OUTPUT_CHUNK_LENGTH={OUTPUT_CHUNK_LENGTH}")
            print("="*50 + "\n")

            # Metrics for PyTorch Lightning logging
            # Some torchmetrics fail due to `.view(-1)` on non-contiguous tensors inside deep Darts architectures.
            # We explicitly define the metric collection kwargs internally or omit it to prevent crashes, since MAE and RMSE are already logged natively by PyTorch Lightning if needed, but we can pass `None` for custom `torch_metrics` and let the PL logger handle it or set a custom `loss_fn`.
            # For safe native scaling, we will omit custom `torch_metrics=metrics.clone()` on models that crash

            # Models definition
            models = get_models(INPUT_CHUNK_LENGTH, OUTPUT_CHUNK_LENGTH)
            true_windows_for_chunk = []

            for name, model in models.items():
                print(f"\nTraining {name}...")

                # Naive models don't use covariates, and don't need training on a whole list
                if name in NAIVE_MODELS:
                    # For naive, we just evaluate on each test block
                    pass # No training required
                elif name in MODELS_WITH_FUTURE_COVARIATES:
                    fit_kwargs = dict(
                        series=train_targets_scaled,
                        val_series=val_targets_scaled,
                    )
                    if has_input_covariates:
                        fit_kwargs["future_covariates"] = train_covariates_scaled
                        fit_kwargs["val_future_covariates"] = val_covariates_scaled
                    if has_past_covariates:
                        fit_kwargs["past_covariates"] = train_past_covariates_scaled
                        fit_kwargs["val_past_covariates"] = val_past_covariates_scaled
                    model.fit(**fit_kwargs)
                elif name in MODELS_FUTURE_COVARIATES_ONLY:
                    fit_kwargs = dict(
                        series=train_targets_scaled,
                        val_series=val_targets_scaled,
                    )
                    if has_input_covariates:
                        fit_kwargs["future_covariates"] = train_covariates_scaled
                        fit_kwargs["val_future_covariates"] = val_covariates_scaled
                    model.fit(**fit_kwargs)
                else:
                    try:
                        model.fit(
                            series=train_targets_scaled,
                            val_series=val_targets_scaled
                        )
                    except Exception as e:
                        print(f"Error training {name}: {e}")

                # Testing
                print(f"Testing {name}...")

                if name in CHECKPOINT_MODELS:
                    try:
                        # Load the best model from checkpoint for testing
                        model_name_for_load = f"{name}_I{INPUT_CHUNK_LENGTH}_O{OUTPUT_CHUNK_LENGTH}"
                        model = type(model).load_from_checkpoint(model_name=model_name_for_load, best=True)
                        print(f"Loaded best checkpoint for {name}.")
                    except Exception as e:
                        print(f"Could not load best checkpoint for {name}: {e}")

                # Calculate training MAE/RMSE (optional, can be very slow for large datasets)
                if name not in NAIVE_MODELS:
                    try:
                        # To keep it quick, we'll just evaluate on a subset or full train
                        selected_indices = [
                            idx for idx, t in enumerate(train_targets_scaled)
                            if len(t) > INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH
                        ]
                        train_series_for_pred = [
                            train_targets_scaled[idx][:-OUTPUT_CHUNK_LENGTH]
                            for idx in selected_indices
                        ]

                        predict_kwargs = dict(
                            n=OUTPUT_CHUNK_LENGTH,
                            series=train_series_for_pred,
                            verbose=False,
                        )
                        if name in MODELS_WITH_FUTURE_COVARIATES:
                            if has_input_covariates:
                                predict_kwargs["future_covariates"] = [train_covariates_scaled[idx] for idx in selected_indices]
                            if has_past_covariates:
                                predict_kwargs["past_covariates"] = [train_past_covariates_scaled[idx] for idx in selected_indices]
                        elif name in MODELS_FUTURE_COVARIATES_ONLY:
                            if has_input_covariates:
                                predict_kwargs["future_covariates"] = [train_covariates_scaled[idx] for idx in selected_indices]

                        train_preds_scaled = model.predict(**predict_kwargs)

                        # Inverse transform the predictions back to original distribution
                        if isinstance(train_preds_scaled, list):
                            train_preds = target_scaler.inverse_transform(train_preds_scaled)
                        else:
                            train_preds = target_scaler.inverse_transform([train_preds_scaled])

                        true_train = [t[-OUTPUT_CHUNK_LENGTH:] for t in train_targets if len(t) > INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH]
                        train_mae = mae(true_train, train_preds)
                        train_rmse = rmse(true_train, train_preds)
                        train_mape = mape(true_train, train_preds)
                        print(f"Training metrics for {name} - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
                    except Exception as e:
                        print(f"Could not calculate train metrics for {name}: {e}")

                mae_list = []
                rmse_list = []
                mape_list = []
                mae_cols = {col: [] for col in TARGET_COLS}
                rmse_cols = {col: [] for col in TARGET_COLS}
                mape_cols = {col: [] for col in TARGET_COLS}
                all_preds = []

                for i, (ts_target, ts_cov, ts_past_cov, ts_target_raw, ts_original_idx) in enumerate(
                    zip(test_targets_scaled, test_covariates_scaled, test_past_covariates_scaled, test_targets_raw, test_series_indices)
                ):
                    # Predict from the last OUTPUT_CHUNK_LENGTH steps
                    # Ensure the series is long enough
                    MAX_INPUT_CHUNK_LENGTH = max(INPUT_CHUNK_LENGTHS)
                    if len(ts_target) <= MAX_INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH:
                        continue

                    stride = 6
                    for forecast_start in range(MAX_INPUT_CHUNK_LENGTH, len(ts_target) - OUTPUT_CHUNK_LENGTH + 1, stride):
                        input_start = forecast_start - INPUT_CHUNK_LENGTH
                        y_train = ts_target[input_start : forecast_start]

                        # y_true needs to be unscaled for metric calculation later
                        y_true = ts_target_raw[forecast_start : forecast_start + OUTPUT_CHUNK_LENGTH]

                        if name in MODELS_WITH_FUTURE_COVARIATES:
                            predict_kwargs = dict(
                                n=OUTPUT_CHUNK_LENGTH,
                                series=y_train,
                                verbose=False,
                            )
                            if has_input_covariates and ts_cov is not None:
                                predict_kwargs["future_covariates"] = ts_cov
                            if has_past_covariates and ts_past_cov is not None:
                                predict_kwargs["past_covariates"] = ts_past_cov
                            pred_scaled = model.predict(**predict_kwargs)
                        elif name in MODELS_FUTURE_COVARIATES_ONLY:
                            predict_kwargs = dict(
                                n=OUTPUT_CHUNK_LENGTH,
                                series=y_train,
                                verbose=False,
                            )
                            if has_input_covariates and ts_cov is not None:
                                predict_kwargs["future_covariates"] = ts_cov
                            pred_scaled = model.predict(**predict_kwargs)
                        elif name in NAIVE_MODELS:
                            model.fit(y_train)
                            pred_scaled = model.predict(n=OUTPUT_CHUNK_LENGTH)
                        else:
                            pred_scaled = model.predict(n=OUTPUT_CHUNK_LENGTH, series=y_train, verbose=False)

                        # Inverse scale predictions directly mapping back real boundaries
                        pred = target_scaler.inverse_transform(pred_scaled)

                        # Reverse optional transforms to return to original scale
                        if use_detrend:
                            pred_retrended = reverse_detrend_timeseries([pred], ts_indices=[ts_original_idx])
                            pred = pred_retrended[0]

                        if use_log_transform:
                            pred_relogged = reverse_log_transform_timeseries([pred], ts_indices=[ts_original_idx])
                            pred = pred_relogged[0]

                        mae_list.append(mae(y_true, pred))
                        rmse_list.append(rmse(y_true, pred))
                        mape_list.append(mape(y_true, pred))

                        for col in TARGET_COLS:
                            mae_cols[col].append(mae(y_true[col], pred[col]))
                            rmse_cols[col].append(rmse(y_true[col], pred[col]))
                            mape_cols[col].append(mape(y_true[col], pred[col]))

                        pred_df = pred.to_dataframe()
                        pred_df['block_idx'] = ts_original_idx
                        pred_df['fcst_origin'] = forecast_start
                        all_preds.append(pred_df)

                        # Save true values only on the first model pass
                        if name == list(models.keys())[0]:
                            true_df = y_true.to_dataframe()
                            true_df['block_idx'] = ts_original_idx
                            true_df['fcst_origin'] = forecast_start
                            true_windows_for_chunk.append(true_df)

                if len(mae_list) > 0:
                    avg_mae = sum(mae_list) / len(mae_list)
                    avg_rmse = sum(rmse_list) / len(rmse_list)
                    avg_mape = sum(mape_list) / len(mape_list)

                    res_dict = {
                        "InputChunkLength": INPUT_CHUNK_LENGTH,
                        "OutputChunkLength": OUTPUT_CHUNK_LENGTH,
                        "Model": name,
                        "MAE": avg_mae,
                        "RMSE": avg_rmse,
                        "MAPE": avg_mape
                    }

                    print(f"Results for {name}: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, MAPE={avg_mape:.4f}")

                    for col in TARGET_COLS:
                        avg_mae_col = sum(mae_cols[col]) / len(mae_cols[col])
                        avg_rmse_col = sum(rmse_cols[col]) / len(rmse_cols[col])
                        avg_mape_col = sum(mape_cols[col]) / len(mape_cols[col])
                        res_dict[f"MAE_{col}"] = avg_mae_col
                        res_dict[f"RMSE_{col}"] = avg_rmse_col
                        res_dict[f"MAPE_{col}"] = avg_mape_col
                        print(f"  {col} - MAE: {avg_mae_col:.4f}, RMSE: {avg_rmse_col:.4f}, MAPE: {avg_mape_col:.4f}")

                    all_results.append(res_dict)

                    model_key = f"{name}_I{INPUT_CHUNK_LENGTH}_O{OUTPUT_CHUNK_LENGTH}"
                    if model_key not in best_models_dict or avg_mae < best_models_dict[model_key]['mae']:
                        best_models_dict[model_key] = {'mae': avg_mae, 'model': model, 'i': INPUT_CHUNK_LENGTH, 'o': OUTPUT_CHUNK_LENGTH, 'name': name}

                    # Save to CSV
                    pd.concat(all_preds).to_csv(f"outputs/pred_{name}_I{INPUT_CHUNK_LENGTH}_O{OUTPUT_CHUNK_LENGTH}.csv")
                    if name == list(models.keys())[0] and true_windows_for_chunk:
                        pd.concat(true_windows_for_chunk).to_csv(f"outputs/true_values_I{INPUT_CHUNK_LENGTH}_O{OUTPUT_CHUNK_LENGTH}.csv")
                else:
                    print(f"No valid test series for {name} (too short)")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = "outputs/evaluation_metrics.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nFinal evaluation metrics saved to {results_path}")

