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

from config import INPUT_COLS, PAST_COLS, TARGET_COLS, TIME_COL, PER_TARGET_CONFIG

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


def load_data(filepath: str, target_cols: list[str], input_cols: list[str], past_cols: list[str]):
    df = pd.read_csv(filepath)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    # A new time series starts when 'session_index' changes
    group_id = (df['session_index'] != df['session_index'].shift()).cumsum()

    targets = []
    covariates = []
    past_covariates = []

    has_input_covariates = len(input_cols) > 0
    has_past_covariates = len(past_cols) > 0

    for _, group_df in df.groupby(group_id):
        group_df = group_df.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL])

        if len(group_df) < 720:
            continue

        # Create targets TimeSeries
        target_ts = TimeSeries.from_dataframe(
            group_df,
            time_col=TIME_COL,
            value_cols=target_cols,
        )
        targets.append(target_ts)

        # Create future covariates TimeSeries
        cov_ts = None
        if has_input_covariates:
            cov_ts = TimeSeries.from_dataframe(
                group_df,
                time_col=TIME_COL,
                value_cols=input_cols,
            )
        covariates.append(cov_ts)

        # Create past covariates TimeSeries
        past_cov_ts = None
        if has_past_covariates:
            past_cov_ts = TimeSeries.from_dataframe(
                group_df,
                time_col=TIME_COL,
                value_cols=past_cols,
            )
        past_covariates.append(past_cov_ts)

    return targets, covariates, past_covariates


def run_training(
    target_cols: list[str],
    input_cols: list[str],
    past_cols: list[str],
    run_tag: str,
    use_detrend: bool,
    use_log_transform: bool,
    shuffle_data: bool,
    max_files_to_load: int | None = None,
):
    has_input_covariates = len(input_cols) > 0
    has_past_covariates = len(past_cols) > 0

    targets_list = []
    covariates_list = []
    past_covariates_list = []

    data_dir = "data_preprocessed"
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    all_files = sorted(all_files)
    if max_files_to_load is not None:
        all_files = all_files[:max_files_to_load]

    for filepath in all_files:
        t, c, p = load_data(filepath, target_cols, input_cols, past_cols)
        targets_list.extend(t)
        covariates_list.extend(c)
        past_covariates_list.extend(p)
        print(f"[{run_tag}] Loaded {len(t)} series blocks from {filepath}")

    print(f"[{run_tag}] Total loaded series blocks: {len(targets_list)}")

    original_targets_list = list(targets_list)
    series_indices = list(range(len(targets_list)))

    if use_log_transform:
        reset_log_transform_storage()
        print(f"\n[{run_tag}] Applying log transform...")
        targets_list = log_transform_timeseries(targets_list)
        if has_input_covariates:
            covariates_list = log_transform_timeseries(covariates_list)
        if has_past_covariates:
            past_covariates_list = log_transform_timeseries(past_covariates_list)
        print(f"[{run_tag}] Log transform completed.")

    if use_detrend:
        reset_detrend_storage()
        print(f"\n[{run_tag}] Detrending data...")
        targets_list = detrend_timeseries_linear(targets_list)
        if has_input_covariates:
            covariates_list = detrend_timeseries_linear(covariates_list)
        if has_past_covariates:
            past_covariates_list = detrend_timeseries_linear(past_covariates_list)
        print(f"[{run_tag}] Data detrending completed.")

    if shuffle_data:
        random.seed(42)
        combined = list(zip(targets_list, covariates_list, past_covariates_list, original_targets_list, series_indices))
        random.shuffle(combined)
        targets_list, covariates_list, past_covariates_list, original_targets_list, series_indices = zip(*combined)

    targets_list = list(targets_list)
    covariates_list = list(covariates_list)
    past_covariates_list = list(past_covariates_list)
    original_targets_list = list(original_targets_list)
    series_indices = list(series_indices)

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

    print(f"[{run_tag}] Train blocks: {len(train_targets)}, Val blocks: {len(val_targets)}, Test blocks: {len(test_targets)}")

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

    input_chunk_lengths = [60, 120]
    output_chunk_lengths = [15, 30, 60]
    all_results = []
    best_models_dict = {}
    true_windows_for_chunk = []

    for input_chunk_length in input_chunk_lengths:
        for output_chunk_length in output_chunk_lengths:
            print(f"\n" + "="*50)
            print(f"[{run_tag}] Training for INPUT_CHUNK_LENGTH={input_chunk_length}, OUTPUT_CHUNK_LENGTH={output_chunk_length}")
            print("="*50 + "\n")

            models = get_models(input_chunk_length, output_chunk_length)
            true_windows_for_chunk = []

            for name, model in models.items():
                print(f"\n[{run_tag}] Training {name}...")

                if name in NAIVE_MODELS:
                    pass
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
                        print(f"[{run_tag}] Error training {name}: {e}")

                print(f"[{run_tag}] Testing {name}...")

                if name in CHECKPOINT_MODELS:
                    try:
                        model_name_for_load = f"{name}_I{input_chunk_length}_O{output_chunk_length}"
                        model = type(model).load_from_checkpoint(model_name=model_name_for_load, best=True)
                        print(f"[{run_tag}] Loaded best checkpoint for {name}.")
                    except Exception as e:
                        print(f"[{run_tag}] Could not load best checkpoint for {name}: {e}")

                if name not in NAIVE_MODELS:
                    try:
                        selected_indices = [
                            idx for idx, t in enumerate(train_targets_scaled)
                            if len(t) > input_chunk_length + output_chunk_length
                        ]
                        train_series_for_pred = [
                            train_targets_scaled[idx][:-output_chunk_length]
                            for idx in selected_indices
                        ]

                        predict_kwargs = dict(
                            n=output_chunk_length,
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

                        if isinstance(train_preds_scaled, list):
                            train_preds = target_scaler.inverse_transform(train_preds_scaled)
                        else:
                            train_preds = target_scaler.inverse_transform([train_preds_scaled])

                        true_train = [t[-output_chunk_length:] for t in train_targets if len(t) > input_chunk_length + output_chunk_length]
                        train_mae = mae(true_train, train_preds)
                        train_rmse = rmse(true_train, train_preds)
                        train_mape = mape(true_train, train_preds)
                        print(f"[{run_tag}] Training metrics for {name} - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
                    except Exception as e:
                        print(f"[{run_tag}] Could not calculate train metrics for {name}: {e}")

                mae_list = []
                rmse_list = []
                mape_list = []
                mae_cols = {col: [] for col in target_cols}
                rmse_cols = {col: [] for col in target_cols}
                mape_cols = {col: [] for col in target_cols}
                all_preds = []

                for ts_target, ts_cov, ts_past_cov, ts_target_raw, ts_original_idx in zip(
                    test_targets_scaled, test_covariates_scaled, test_past_covariates_scaled, test_targets_raw, test_series_indices
                ):
                    max_input_chunk_length = max(input_chunk_lengths)
                    if len(ts_target) <= max_input_chunk_length + output_chunk_length:
                        continue

                    stride = 6
                    for forecast_start in range(max_input_chunk_length, len(ts_target) - output_chunk_length + 1, stride):
                        input_start = forecast_start - input_chunk_length
                        y_train = ts_target[input_start: forecast_start]
                        y_true = ts_target_raw[forecast_start: forecast_start + output_chunk_length]

                        if name in MODELS_WITH_FUTURE_COVARIATES:
                            predict_kwargs = dict(
                                n=output_chunk_length,
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
                                n=output_chunk_length,
                                series=y_train,
                                verbose=False,
                            )
                            if has_input_covariates and ts_cov is not None:
                                predict_kwargs["future_covariates"] = ts_cov
                            pred_scaled = model.predict(**predict_kwargs)
                        elif name in NAIVE_MODELS:
                            model.fit(y_train)
                            pred_scaled = model.predict(n=output_chunk_length)
                        else:
                            pred_scaled = model.predict(n=output_chunk_length, series=y_train, verbose=False)

                        pred = target_scaler.inverse_transform(pred_scaled)

                        if use_detrend:
                            pred_retrended = reverse_detrend_timeseries([pred], ts_indices=[ts_original_idx])
                            pred = pred_retrended[0]

                        if use_log_transform:
                            pred_relogged = reverse_log_transform_timeseries([pred], ts_indices=[ts_original_idx])
                            pred = pred_relogged[0]

                        mae_list.append(mae(y_true, pred))
                        rmse_list.append(rmse(y_true, pred))
                        mape_list.append(mape(y_true, pred))

                        for col in target_cols:
                            mae_cols[col].append(mae(y_true[col], pred[col]))
                            rmse_cols[col].append(rmse(y_true[col], pred[col]))
                            mape_cols[col].append(mape(y_true[col], pred[col]))

                        pred_df = pred.to_dataframe()
                        pred_df['block_idx'] = ts_original_idx
                        pred_df['fcst_origin'] = forecast_start
                        all_preds.append(pred_df)

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
                        "RunTag": run_tag,
                        "InputChunkLength": input_chunk_length,
                        "OutputChunkLength": output_chunk_length,
                        "Model": name,
                        "MAE": avg_mae,
                        "RMSE": avg_rmse,
                        "MAPE": avg_mape,
                    }

                    print(f"[{run_tag}] Results for {name}: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, MAPE={avg_mape:.4f}")

                    for col in target_cols:
                        avg_mae_col = sum(mae_cols[col]) / len(mae_cols[col])
                        avg_rmse_col = sum(rmse_cols[col]) / len(rmse_cols[col])
                        avg_mape_col = sum(mape_cols[col]) / len(mape_cols[col])
                        res_dict[f"MAE_{col}"] = avg_mae_col
                        res_dict[f"RMSE_{col}"] = avg_rmse_col
                        res_dict[f"MAPE_{col}"] = avg_mape_col
                        print(f"[{run_tag}]   {col} - MAE: {avg_mae_col:.4f}, RMSE: {avg_rmse_col:.4f}, MAPE: {avg_mape_col:.4f}")

                    all_results.append(res_dict)

                    model_key = f"{name}_I{input_chunk_length}_O{output_chunk_length}"
                    if model_key not in best_models_dict or avg_mae < best_models_dict[model_key]['mae']:
                        best_models_dict[model_key] = {
                            'mae': avg_mae,
                            'model': model,
                            'i': input_chunk_length,
                            'o': output_chunk_length,
                            'name': name,
                        }

                    preds_path = f"outputs/pred_{run_tag}_{name}_I{input_chunk_length}_O{output_chunk_length}.csv"
                    pd.concat(all_preds).to_csv(preds_path)
                    if name == list(models.keys())[0] and true_windows_for_chunk:
                        true_path = f"outputs/true_values_{run_tag}_I{input_chunk_length}_O{output_chunk_length}.csv"
                        pd.concat(true_windows_for_chunk).to_csv(true_path)
                else:
                    print(f"[{run_tag}] No valid test series for {name} (too short)")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = f"outputs/evaluation_metrics_{run_tag}.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n[{run_tag}] Final evaluation metrics saved to {results_path}")

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

    print(f"use_detrend={use_detrend}, use_log_transform={use_log_transform}")

    os.makedirs("outputs", exist_ok=True)

    max_files_to_load = None

    run_training(
        target_cols=TARGET_COLS,
        input_cols=INPUT_COLS,
        past_cols=PAST_COLS,
        run_tag="all_targets",
        use_detrend=use_detrend,
        use_log_transform=use_log_transform,
        shuffle_data=shuffle_data,
        max_files_to_load=max_files_to_load,
    )

    for target, cfg in PER_TARGET_CONFIG.items():
        run_training(
            target_cols=[target],
            input_cols=cfg.get("input", []),
            past_cols=cfg.get("past", []),
            run_tag=f"target_{target}",
            use_detrend=use_detrend,
            use_log_transform=use_log_transform,
            shuffle_data=shuffle_data,
            max_files_to_load=max_files_to_load,
        )

