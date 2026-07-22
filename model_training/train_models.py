import pandas as pd
import argparse
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, rmse, mape
import sys
import os
import glob
import random
import warnings
import logging
import pickle
from pathlib import Path
import numpy as np
from sklearn.preprocessing import PowerTransformer

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)
logging.getLogger("darts").setLevel(logging.ERROR)

# Allow running this file directly (e.g. `python model_training/train_models.py`)
# by ensuring the project root (where `config.py` lives) is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import INPUT_COLS, PAST_COLS, TARGET_COLS, TIME_COL, PER_TARGET_CONFIG
from config import SIMPLE_MODEL_CONFIG
try:
    from config import BOUNDS
except ImportError:
    BOUNDS = {}

from model_training.detrend_data import (
    detrend_timeseries_linear,
    reverse_detrend_timeseries,
    reset_detrend_storage,
)
from model_training.log_transform_data import (
    log_transform_timeseries,
    reverse_log_transform_timeseries,
    reset_log_transform_storage,
)
from model_training.model_definitions import (
    get_models,
    NAIVE_MODELS,
    MODELS_WITH_FUTURE_COVARIATES,
    MODELS_FUTURE_COVARIATES_ONLY,
    CHECKPOINT_MODELS,
)


input_chunk_lengths = [2, 8, 6, 8]
output_chunk_lengths = [1, 2, 3]

min_series_length = max(input_chunk_lengths) + max(output_chunk_lengths)
print(min_series_length)

_MODEL_GROUPS: dict[str, set[str]] = {
    "naive": set(NAIVE_MODELS),
    "checkpoint": set(CHECKPOINT_MODELS),
    "future_covariates": set(MODELS_WITH_FUTURE_COVARIATES),
    "future_only": set(MODELS_FUTURE_COVARIATES_ONLY),
}


def _parse_csv_arg(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return set(items) if items else None


def _filter_models(
    models: dict,
    *,
    include_names: set[str] | None,
    exclude_names: set[str] | None,
    include_groups: set[str] | None,
) -> dict:
    filtered = dict(models)

    if include_groups:
        known_union = set().union(*_MODEL_GROUPS.values())
        selected: set[str] = set()
        for g in include_groups:
            if g == "other":
                selected |= set(filtered.keys()) - (known_union & set(filtered.keys()))
            else:
                selected |= _MODEL_GROUPS[g] & set(filtered.keys())
        filtered = {k: v for k, v in filtered.items() if k in selected}

    if include_names:
        filtered = {k: v for k, v in filtered.items() if k in include_names}

    if exclude_names:
        filtered = {k: v for k, v in filtered.items() if k not in exclude_names}

    return filtered


def apply_exponential_smoothing(ts_list: list, alpha: float) -> list:
    """Applies Exponential Smoothing to a list of Darts TimeSeries."""
    smoothed = []
    for ts in ts_list:
        if ts is None:
            smoothed.append(None)
        else:
            df = ts.to_dataframe()
            df_smoothed = df.ewm(alpha=alpha, adjust=False).mean()
            smoothed.append(TimeSeries.from_dataframe(df_smoothed, fill_missing_dates=True, freq=None))
    return smoothed


def load_data(
    filepath: str,
    target_cols: list[str],
    input_cols: list[str],
    past_cols: list[str],
):
    df = pd.read_csv(filepath)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    all_numeric_cols = set(target_cols + input_cols + past_cols)
    for col in all_numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    # A new time series starts when 'session_index' changes
    group_id = (df['session_index'] != df['session_index'].shift()).cumsum()

    targets = []
    covariates = []
    past_covariates = []

    has_input_covariates = len(input_cols) > 0
    has_past_covariates = len(past_cols) > 0

    for _, group_df in df.groupby(group_id):
        group_df = group_df.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL])

        if len(group_df) <= min_series_length:
            continue

        # Create targets TimeSeries
        target_ts = TimeSeries.from_dataframe(
            group_df,
            time_col=TIME_COL,
            value_cols=target_cols,
            fill_missing_dates=True,
            freq=None
        )
        targets.append(target_ts)

        # Create future covariates TimeSeries
        cov_ts = None
        if has_input_covariates:
            cov_ts = TimeSeries.from_dataframe(
                group_df,
                time_col=TIME_COL,
                value_cols=input_cols,
                fill_missing_dates=True,
                freq=None
            )
        covariates.append(cov_ts)

        # Create past covariates TimeSeries
        past_cov_ts = None
        if has_past_covariates:
            past_cov_ts = TimeSeries.from_dataframe(
                group_df,
                time_col=TIME_COL,
                value_cols=past_cols,
                fill_missing_dates=True,
                freq=None
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
    use_box_cox: bool,
    shuffle_data: bool,
    exp_smoothing_alpha: float | None = None,
    eval_on_smoothed: bool = False,
    model_names: set[str] | None = None,
    exclude_model_names: set[str] | None = None,
    model_groups: set[str] | None = None,
    max_files_to_load: int | None = None,
    artifact_group: str | None = None,
    use_builtin_scalers: bool = False,
    split_per_session: bool = False,
    disable_mape: bool = False,
    clip_predictions: bool = False,
):
    artifact_group = artifact_group or run_tag
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
        t, c, p = load_data(
            filepath,
            target_cols,
            input_cols,
            past_cols,
        )
        targets_list.extend(t)
        covariates_list.extend(c)
        past_covariates_list.extend(p)
        print(f"[{run_tag}] Loaded {len(t)} series blocks from {filepath}")

    print(f"[{run_tag}] Total loaded series blocks: {len(targets_list)}")

    # Capture original lists BEFORE any smoothing or transformation
    original_targets_list = list(targets_list)
    series_indices = list(range(len(targets_list)))

    # Apply Exponential Smoothing if requested
    if exp_smoothing_alpha is not None:
        print(f"\n[{run_tag}] Applying exponential smoothing (alpha={exp_smoothing_alpha})...")
        targets_list = apply_exponential_smoothing(targets_list, exp_smoothing_alpha)
        if has_input_covariates:
            covariates_list = apply_exponential_smoothing(covariates_list, exp_smoothing_alpha)
        if has_past_covariates:
            past_covariates_list = apply_exponential_smoothing(past_covariates_list, exp_smoothing_alpha)
        print(f"[{run_tag}] Exponential smoothing completed.")

    # Capture state after smoothing but BEFORE log/detrend for evaluation if requested
    smoothed_targets_list = list(targets_list)

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
        combined = list(zip(
            targets_list,
            covariates_list,
            past_covariates_list,
            original_targets_list,
            smoothed_targets_list,
            series_indices
        ))
        random.shuffle(combined)
        (targets_list,
         covariates_list,
         past_covariates_list,
         original_targets_list,
         smoothed_targets_list,
         series_indices) = zip(*combined)

    targets_list = list(targets_list)
    covariates_list = list(covariates_list)
    past_covariates_list = list(past_covariates_list)
    original_targets_list = list(original_targets_list)
    smoothed_targets_list = list(smoothed_targets_list)
    series_indices = list(series_indices)

    n_total = len(targets_list)

    if split_per_session:
        print(f"[{run_tag}] Splitting EACH session temporally into Train/Val/Test...")
        train_targets, val_targets, test_targets = [], [], []
        train_covariates, val_covariates, test_covariates = [], [], []
        train_past_covariates, val_past_covariates, test_past_covariates = [], [], []
        test_series_indices = []
        test_targets_raw = []

        for i in range(n_total):
            ts = targets_list[i]
            L = len(ts)
            s1 = int(L * 0.7)
            s2 = int(L * 0.85)

            if s1 == 0 or s2 == s1 or s2 == L:
                continue

            train_targets.append(ts[:s1])
            val_targets.append(ts[s1:s2])
            test_targets.append(ts[s2:])

            if covariates_list[i] is not None:
                train_covariates.append(covariates_list[i][:s1])
                val_covariates.append(covariates_list[i][s1:s2])
                test_covariates.append(covariates_list[i][s2:])
            else:
                train_covariates.append(None)
                val_covariates.append(None)
                test_covariates.append(None)

            if past_covariates_list[i] is not None:
                train_past_covariates.append(past_covariates_list[i][:s1])
                val_past_covariates.append(past_covariates_list[i][s1:s2])
                test_past_covariates.append(past_covariates_list[i][s2:])
            else:
                train_past_covariates.append(None)
                val_past_covariates.append(None)
                test_past_covariates.append(None)

            test_series_indices.append(series_indices[i])

            if eval_on_smoothed:
                test_targets_raw.append(smoothed_targets_list[i][s2:])
            else:
                test_targets_raw.append(original_targets_list[i][s2:])

        if eval_on_smoothed:
             print(f"[{run_tag}] Metrics will be calculated against SMOOTHED ground truth.")
        else:
             print(f"[{run_tag}] Metrics will be calculated against ORIGINAL RAW ground truth.")

    else:
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
        test_series_indices = series_indices[split_idx_2:]

        if eval_on_smoothed:
            test_targets_raw = smoothed_targets_list[split_idx_2:]
            print(f"[{run_tag}] Metrics will be calculated against SMOOTHED ground truth.")
        else:
            test_targets_raw = original_targets_list[split_idx_2:]
            print(f"[{run_tag}] Metrics will be calculated against ORIGINAL RAW ground truth.")

    print(f"[{run_tag}] Before global filtering - Train blocks: {len(train_targets)}, Val blocks: {len(val_targets)}, Test blocks: {len(test_targets)}")

    # =========================================================================================
    # GLOBAL LENGTH FILTERING TO GUARANTEE EXACT SAME DATASET ACROSS ALL CONFIGURATIONS/MODELS
    # =========================================================================================
    max_req_len = max(input_chunk_lengths) + max(output_chunk_lengths)

    def _filter_by_length(targets, covs, past_covs, raw_targets=None, indices=None):
        valid = [i for i, t in enumerate(targets) if len(t) >= max_req_len]
        res_targets = [targets[i] for i in valid]
        res_covs = [covs[i] for i in valid] if covs else []
        res_past_covs = [past_covs[i] for i in valid] if past_covs else []
        res_raw = [raw_targets[i] for i in valid] if raw_targets is not None else None
        res_idx = [indices[i] for i in valid] if indices is not None else None
        return res_targets, res_covs, res_past_covs, res_raw, res_idx

    train_targets, train_covariates, train_past_covariates, _, _ = _filter_by_length(
        train_targets, train_covariates, train_past_covariates
    )
    val_targets, val_covariates, val_past_covariates, _, _ = _filter_by_length(
        val_targets, val_covariates, val_past_covariates
    )
    test_targets, test_covariates, test_past_covariates, test_targets_raw, test_series_indices = _filter_by_length(
        test_targets, test_covariates, test_past_covariates, test_targets_raw, test_series_indices
    )

    if not train_targets:
        print(f"[{run_tag}] FATAL: No training series left after length filtering (min req length {max_req_len}). Exiting run.")
        return

    print(f"[{run_tag}] After global filtering (min length {max_req_len}) - Train: {len(train_targets)}, Val: {len(val_targets)}, Test: {len(test_targets)}")
    # =========================================================================================

    if use_box_cox:
        target_boxcox = Scaler(scaler=PowerTransformer(method='yeo-johnson'), global_fit=True)
        covariates_boxcox = Scaler(scaler=PowerTransformer(method='yeo-johnson'), global_fit=True) if has_input_covariates else None
        past_covariates_boxcox = Scaler(scaler=PowerTransformer(method='yeo-johnson'), global_fit=True) if has_past_covariates else None

        train_targets = target_boxcox.fit_transform(train_targets)
        train_covariates = covariates_boxcox.fit_transform(train_covariates) if has_input_covariates else [None] * len(train_targets)
        train_past_covariates = past_covariates_boxcox.fit_transform(train_past_covariates) if has_past_covariates else [None] * len(train_targets)

        val_targets = target_boxcox.transform(val_targets) if val_targets else []
        val_covariates = covariates_boxcox.transform(val_covariates) if has_input_covariates and val_targets else [None] * len(val_targets)
        val_past_covariates = past_covariates_boxcox.transform(val_past_covariates) if has_past_covariates and val_targets else [None] * len(val_targets)

        test_targets = target_boxcox.transform(test_targets) if test_targets else []
        test_covariates = covariates_boxcox.transform(test_covariates) if has_input_covariates and test_targets else [None] * len(test_targets)
        test_past_covariates = past_covariates_boxcox.transform(test_past_covariates) if has_past_covariates and test_targets else [None] * len(test_targets)
    else:
        target_boxcox, covariates_boxcox, past_covariates_boxcox = None, None, None


    if not use_builtin_scalers:
        target_scaler = Scaler(global_fit=True)
        covariates_scaler = Scaler(global_fit=True) if has_input_covariates else None
        past_covariates_scaler = Scaler(global_fit=True) if has_past_covariates else None

        train_targets_scaled = target_scaler.fit_transform(train_targets)
        train_covariates_scaled = covariates_scaler.fit_transform(train_covariates) if has_input_covariates else [None] * len(train_targets)
        train_past_covariates_scaled = past_covariates_scaler.fit_transform(train_past_covariates) if has_past_covariates else [None] * len(train_targets)

        val_targets_scaled = target_scaler.transform(val_targets) if val_targets else []
        val_covariates_scaled = covariates_scaler.transform(val_covariates) if has_input_covariates and val_targets else [None] * len(val_targets)
        val_past_covariates_scaled = past_covariates_scaler.transform(val_past_covariates) if has_past_covariates and val_targets else [None] * len(val_targets)

        test_targets_scaled = target_scaler.transform(test_targets) if test_targets else []
        test_covariates_scaled = covariates_scaler.transform(test_covariates) if has_input_covariates and test_targets else [None] * len(test_targets)
        test_past_covariates_scaled = past_covariates_scaler.transform(test_past_covariates) if has_past_covariates and test_targets else [None] * len(test_targets)
    else:
        target_scaler, covariates_scaler, past_covariates_scaler = None, None, None
        train_targets_scaled = train_targets
        train_covariates_scaled = train_covariates if has_input_covariates else [None] * len(train_targets)
        train_past_covariates_scaled = train_past_covariates if has_past_covariates else [None] * len(train_targets)

        val_targets_scaled = val_targets
        val_covariates_scaled = val_covariates if has_input_covariates else [None] * len(val_targets)
        val_past_covariates_scaled = val_past_covariates if has_past_covariates else [None] * len(val_targets)

        test_targets_scaled = test_targets
        test_covariates_scaled = test_covariates if has_input_covariates else [None] * len(test_targets)
        test_past_covariates_scaled = test_past_covariates if has_past_covariates else [None] * len(test_targets)

    def persist_scalers(model_name: str) -> None:
        """Save scalers next to the Darts checkpoint folder."""
        if use_builtin_scalers:
            return
        out_path = Path("darts_logs") / model_name / "scalers.pkl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "target_scaler": target_scaler,
            "target_boxcox": target_boxcox,
            "covariates_scaler": covariates_scaler,
            "covariates_boxcox": covariates_boxcox,
            "past_covariates_scaler": past_covariates_scaler,
            "past_covariates_boxcox": past_covariates_boxcox,
            "meta": {
                "target_cols": list(target_cols),
                "input_cols": list(input_cols),
                "past_cols": list(past_cols),
                "run_tag": run_tag,
            },
        }
        with out_path.open("wb") as f:
            pickle.dump(bundle, f)

    def persist_serialized_model(model_name: str, model_obj) -> None:
        """Save a non-checkpoint Darts model next to `darts_logs/<model_name>/`."""
        out_dir = Path("darts_logs") / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "_model.pth.tar"

        if not hasattr(model_obj, "save"):
            raise AttributeError(f"Model {type(model_obj)} has no .save() method")

        model_obj.save(str(out_path))

    all_results = []
    best_models_dict = {}
    true_windows_for_chunk = []

    for input_chunk_length in input_chunk_lengths:
        for output_chunk_length in output_chunk_lengths:
            print(f"\n" + "="*50)
            print(f"[{run_tag}] Training for INPUT_CHUNK_LENGTH={input_chunk_length}, OUTPUT_CHUNK_LENGTH={output_chunk_length}")
            print("="*50 + "\n")

            # Point to globally filtered lists
            model_train_targets = train_targets_scaled
            model_train_covs = train_covariates_scaled
            model_train_past_covs = train_past_covariates_scaled

            model_val_targets = val_targets_scaled if val_targets_scaled else None
            model_val_covs = val_covariates_scaled if val_targets_scaled else None
            model_val_past_covs = val_past_covariates_scaled if val_targets_scaled else None

            model_test_targets = test_targets_scaled

            models = get_models(
                input_chunk_length,
                output_chunk_length,
                input_cols=input_cols,
                past_cols=past_cols,
                run_group=artifact_group,
                use_builtin_scalers=use_builtin_scalers,
            )
            models = _filter_models(
                models,
                include_names=model_names,
                exclude_names=exclude_model_names,
                include_groups=model_groups,
            )
            if not models:
                print(
                    f"[{run_tag}] No models selected for I{input_chunk_length}/O{output_chunk_length} (filter removed all)."
                )
                continue
            true_windows_for_chunk = []

            for name, model in models.items():
                print(f"\n[{run_tag}] Training {name}...")

                model_name_for_artifacts = f"{name}_I{input_chunk_length}_O{output_chunk_length}"

                if name in CHECKPOINT_MODELS:
                    persist_scalers(model_name_for_artifacts)
                elif name == "LinearRegression" or name == "Chronos2":
                    persist_scalers(model_name_for_artifacts)

                if name in NAIVE_MODELS:
                    pass
                elif name in MODELS_WITH_FUTURE_COVARIATES:
                    fit_kwargs = dict(series=model_train_targets)
                    if model_val_targets is not None:
                        fit_kwargs["val_series"] = model_val_targets

                    if has_input_covariates:
                        fit_kwargs["future_covariates"] = model_train_covs
                        if model_val_targets is not None:
                            fit_kwargs["val_future_covariates"] = model_val_covs
                    if has_past_covariates:
                        fit_kwargs["past_covariates"] = model_train_past_covs
                        if model_val_targets is not None:
                            fit_kwargs["val_past_covariates"] = model_val_past_covs
                    if name == "Chronos2":
                        fit_kwargs["epochs"] = 0
                    model.fit(**fit_kwargs)
                elif name in MODELS_FUTURE_COVARIATES_ONLY:
                    fit_kwargs = dict(series=model_train_targets)
                    if model_val_targets is not None:
                        fit_kwargs["val_series"] = model_val_targets

                    if has_input_covariates:
                        fit_kwargs["future_covariates"] = model_train_covs
                        if model_val_targets is not None:
                            fit_kwargs["val_future_covariates"] = model_val_covs
                    model.fit(**fit_kwargs)
                else:
                    try:
                        fit_kwargs = dict(series=model_train_targets)
                        if model_val_targets is not None:
                            fit_kwargs["val_series"] = model_val_targets
                        model.fit(**fit_kwargs)
                    except Exception as e:
                        print(f"[{run_tag}] Error training {name}: {e}")

                print(f"[{run_tag}] Testing {name}...")

                if name in CHECKPOINT_MODELS:
                    try:
                        model = type(model).load_from_checkpoint(model_name=model_name_for_artifacts, best=True)
                        print(f"[{run_tag}] Loaded best checkpoint for {name}.")
                    except Exception as e:
                        print(f"[{run_tag}] Could not load best checkpoint for {name}: {e}")
                elif name == "LinearRegression" or name == "Chronos2":
                    try:
                        persist_serialized_model(model_name_for_artifacts, model)
                        print(f"[{run_tag}] Saved serialized model for {name} to darts_logs/{model_name_for_artifacts}/_model.pth.tar")
                    except Exception as e:
                        print(f"[{run_tag}] Could not save serialized model for {name}: {e}")

                if name not in NAIVE_MODELS:
                    try:
                        train_series_for_pred = [
                            model_train_targets[idx][:-output_chunk_length]
                            for idx in range(len(model_train_targets))
                        ]

                        predict_kwargs = dict(
                            n=output_chunk_length,
                            series=train_series_for_pred,
                            verbose=False,
                        )
                        if name in MODELS_WITH_FUTURE_COVARIATES:
                            if has_input_covariates:
                                predict_kwargs["future_covariates"] = [model_train_covs[idx] for idx in range(len(model_train_targets))]
                            if has_past_covariates:
                                predict_kwargs["past_covariates"] = [model_train_past_covs[idx] for idx in range(len(model_train_targets))]
                        elif name in MODELS_FUTURE_COVARIATES_ONLY:
                            if has_input_covariates:
                                predict_kwargs["future_covariates"] = [model_train_covs[idx] for idx in range(len(model_train_targets))]

                        train_preds_scaled = model.predict(**predict_kwargs)

                        if not use_builtin_scalers:
                            if isinstance(train_preds_scaled, list):
                                train_preds = target_scaler.inverse_transform(train_preds_scaled)
                            else:
                                train_preds = target_scaler.inverse_transform([train_preds_scaled])
                        else:
                            train_preds = train_preds_scaled if isinstance(train_preds_scaled, list) else [train_preds_scaled]

                        if use_box_cox:
                            if isinstance(train_preds, list):
                                train_preds = target_boxcox.inverse_transform(train_preds)
                            else:
                                train_preds = target_boxcox.inverse_transform([train_preds])
                        true_train = [model_train_targets[idx][-output_chunk_length:] for idx in range(len(model_train_targets))]
                        if not use_builtin_scalers:
                            true_train = target_scaler.inverse_transform(true_train)

                        if use_box_cox:
                            true_train = target_boxcox.inverse_transform(true_train)

                        train_mae = mae(true_train, train_preds)
                        train_rmse = rmse(true_train, train_preds)

                        if not disable_mape:
                            train_mape = mape(true_train, train_preds)
                            print(f"[{run_tag}] Training metrics for {name} - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
                        else:
                            print(f"[{run_tag}] Training metrics for {name} - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")

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
                    model_test_targets, test_covariates_scaled, test_past_covariates_scaled, test_targets_raw, test_series_indices
                ):
                    stride = 1
                    max_input_chunk_length = max(input_chunk_lengths)
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

                        if not use_builtin_scalers:
                            pred = target_scaler.inverse_transform(pred_scaled)
                        else:
                            pred = pred_scaled

                        if use_box_cox:
                            pred = target_boxcox.inverse_transform([pred])[0]

                        if use_detrend:
                            pred_retrended = reverse_detrend_timeseries([pred], ts_indices=[ts_original_idx])
                            pred = pred_retrended[0]

                        if use_log_transform:
                            pred_relogged = reverse_log_transform_timeseries([pred], ts_indices=[ts_original_idx])
                            pred = pred_relogged[0]

                        # --- APPLY CLIPPING BASED ON BOUNDS DICTIONARY ---
                        if clip_predictions:
                            pred_df = pred.to_dataframe()
                            for col in pred_df.columns:
                                if col in BOUNDS:
                                    c_min = BOUNDS[col].get("min", None)
                                    c_max = BOUNDS[col].get("max", None)
                                    pred_df[col] = pred_df[col].clip(lower=c_min, upper=c_max)
                            pred = pred.with_values(pred_df.values)

                        mae_list.append(mae(y_true, pred))
                        rmse_list.append(rmse(y_true, pred))
                        if not disable_mape:
                            mape_list.append(mape(y_true, pred))

                        for col in target_cols:
                            mae_cols[col].append(mae(y_true[col], pred[col]))
                            rmse_cols[col].append(rmse(y_true[col], pred[col]))
                            if not disable_mape:
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

                    res_dict = {
                        "RunTag": run_tag,
                        "InputChunkLength": input_chunk_length,
                        "OutputChunkLength": output_chunk_length,
                        "Model": name,
                        "MAE": avg_mae,
                        "RMSE": avg_rmse,
                    }

                    if not disable_mape:
                        avg_mape = sum(mape_list) / len(mape_list)
                        res_dict["MAPE"] = avg_mape
                        print(f"[{run_tag}] Results for {name}: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, MAPE={avg_mape:.4f}")
                    else:
                        print(f"[{run_tag}] Results for {name}: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}")

                    for col in target_cols:
                        avg_mae_col = sum(mae_cols[col]) / len(mae_cols[col])
                        avg_rmse_col = sum(rmse_cols[col]) / len(rmse_cols[col])
                        res_dict[f"MAE_{col}"] = avg_mae_col
                        res_dict[f"RMSE_{col}"] = avg_rmse_col

                        if not disable_mape:
                            avg_mape_col = sum(mape_cols[col]) / len(mape_cols[col])
                            res_dict[f"MAPE_{col}"] = avg_mape_col
                            print(f"[{run_tag}]   {col} - MAE: {avg_mae_col:.4f}, RMSE: {avg_rmse_col:.4f}, MAPE: {avg_mape_col:.4f}")
                        else:
                            print(f"[{run_tag}]   {col} - MAE: {avg_mae_col:.4f}, RMSE: {avg_rmse_col:.4f}")

                    all_results.append(res_dict)

                    model_key = f"{name}_I{input_chunk_length}_O{output_chunk_length}"
                    score_to_track = avg_mape if not disable_mape else avg_mae

                    if model_key not in best_models_dict or score_to_track < best_models_dict[model_key]['score']:
                        best_models_dict[model_key] = {
                            'score': score_to_track,
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
        "--use-box-cox",
        action="store_true",
        help="Apply PowerTransformer (Yeo-Johnson) before training and reverse it after prediction.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the data blocks before train/val/test split.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=["all", "all_targets", "per_target", "simple"],
        default=["all"],
        help="Which training runs to execute (default: all).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model names to train (e.g. 'LinearRegression,NeuralForecast_Nhits').",
    )
    parser.add_argument(
        "--exclude-models",
        type=str,
        default=None,
        help="Comma-separated list of model names to exclude.",
    )
    parser.add_argument(
        "--model-groups",
        nargs="+",
        choices=["naive", "checkpoint", "future_covariates", "future_only", "other"],
        default=None,
        help="Train only selected model groups.",
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=0,
        help="If > 0, calculates moving average column names. Target MAs are added to PAST_COLS to prevent data leakage.",
    )
    parser.add_argument(
        "--exp-smoothing-alpha",
        type=float,
        default=None,
        help="Apply exponential smoothing with the given alpha (0 < alpha <= 1) to all variables before training.",
    )
    parser.add_argument(
        "--eval-on-smoothed",
        action="store_true",
        help="If passed, evaluation metrics (MAE, RMSE, MAPE) are calculated against the smoothed data instead of the raw data.",
    )
    parser.add_argument(
        "--use-builtin-scalers",
        action="store_true",
        help="Skip external scalers and use models' built-in scalers (robust_statistics).",
    )
    parser.add_argument(
        "--split-per-session",
        action="store_true",
        help="Divide each session temporally into train (70%), val (15%), and test (15%) instead of splitting the entire list of sessions.",
    )
    parser.add_argument(
        "--disable-mape",
        action="store_true",
        help="Disable the MAPE metric calculation (useful if encountering zero-division warnings/errors).",
    )
    parser.add_argument(
        "--clip-predictions",
        action="store_true",
        help="Clip predictions on the test set based on the BOUNDS defined in config.py.",
    )
    args = parser.parse_args()

    # Apply MA Window configurations if requested
    if args.ma_window > 0:
        w = args.ma_window

        target_ma_cols = [f"{c}_ma_{w}" for c in TARGET_COLS]
        input_ma_cols = [f"{c}_ma_{w}" for c in INPUT_COLS]
        past_ma_cols = [f"{c}_ma_{w}" for c in PAST_COLS]

        # Target columns remain exactly the same (do not predict the moving average)
        # Inputs get their own MAs
        INPUT_COLS = list(INPUT_COLS) + input_ma_cols

        # Past Covariates get their own MAs PLUS the moving averages of the targets
        PAST_COLS = list(PAST_COLS) + past_ma_cols + target_ma_cols

        # Update per target config in place
        for target_key, cfg in PER_TARGET_CONFIG.items():
            if "input" in cfg:
                cfg["input"] = list(cfg["input"]) + [f"{c}_ma_{w}" for c in cfg["input"]]

            # Append past MAs and the current target's MA to past covariates
            if "past" in cfg:
                cfg["past"] = list(cfg["past"]) + [f"{c}_ma_{w}" for c in cfg["past"]] + [f"{target_key}_ma_{w}"]
            else:
                cfg["past"] = [f"{target_key}_ma_{w}"]

        # Update simple model config in place
        simple_target_cols = SIMPLE_MODEL_CONFIG.get("target_cols", TARGET_COLS)
        simple_target_mas = [f"{c}_ma_{w}" for c in simple_target_cols]

        if "input_cols" in SIMPLE_MODEL_CONFIG:
            SIMPLE_MODEL_CONFIG["input_cols"] = list(SIMPLE_MODEL_CONFIG["input_cols"]) + [f"{c}_ma_{w}" for c in SIMPLE_MODEL_CONFIG["input_cols"]]
        elif "input" in SIMPLE_MODEL_CONFIG:
            SIMPLE_MODEL_CONFIG["input"] = list(SIMPLE_MODEL_CONFIG["input"]) + [f"{c}_ma_{w}" for c in SIMPLE_MODEL_CONFIG["input"]]

        if "past_cols" in SIMPLE_MODEL_CONFIG:
            SIMPLE_MODEL_CONFIG["past_cols"] = list(SIMPLE_MODEL_CONFIG["past_cols"]) + [f"{c}_ma_{w}" for c in SIMPLE_MODEL_CONFIG["past_cols"]] + simple_target_mas
        elif "past" in SIMPLE_MODEL_CONFIG:
            SIMPLE_MODEL_CONFIG["past"] = list(SIMPLE_MODEL_CONFIG["past"]) + [f"{c}_ma_{w}" for c in SIMPLE_MODEL_CONFIG["past"]] + simple_target_mas
        else:
            # If neither exist, initialize past_cols with target moving averages
            SIMPLE_MODEL_CONFIG["past_cols"] = simple_target_mas

    use_detrend = args.use_detrend
    use_log_transform = args.use_log_transform
    use_box_cox = args.use_box_cox
    shuffle_data = args.shuffle
    use_builtin_scalers = args.use_builtin_scalers
    disable_mape = args.disable_mape
    clip_predictions = args.clip_predictions

    print(f"use_detrend={use_detrend}, use_log_transform={use_log_transform}, use_box_cox={use_box_cox}, ma_window={args.ma_window}, exp_smoothing_alpha={args.exp_smoothing_alpha}, eval_on_smoothed={args.eval_on_smoothed}, use_builtin_scalers={use_builtin_scalers}, split_per_session={args.split_per_session}, disable_mape={disable_mape}, clip_predictions={clip_predictions}")

    os.makedirs("outputs", exist_ok=True)

    max_files_to_load = None

    runs = set(args.runs)
    if "all" in runs:
        runs = {"all_targets", "per_target", "simple"}

    model_names = _parse_csv_arg(args.models)
    exclude_model_names = _parse_csv_arg(args.exclude_models)
    model_groups = set(args.model_groups) if args.model_groups else None

    if "all_targets" in runs:
        run_training(
            target_cols=TARGET_COLS,
            input_cols=INPUT_COLS,
            past_cols=PAST_COLS,
            run_tag="all_targets",
            artifact_group="all_targets",
            use_detrend=use_detrend,
            use_log_transform=use_log_transform,
            use_box_cox=use_box_cox,
            shuffle_data=shuffle_data,
            exp_smoothing_alpha=args.exp_smoothing_alpha,
            eval_on_smoothed=args.eval_on_smoothed,
            model_names=model_names,
            exclude_model_names=exclude_model_names,
            model_groups=model_groups,
            max_files_to_load=max_files_to_load,
            use_builtin_scalers=use_builtin_scalers,
            split_per_session=args.split_per_session,
            disable_mape=disable_mape,
            clip_predictions=clip_predictions,
        )

    if "per_target" in runs:
        for target, cfg in PER_TARGET_CONFIG.items():
            run_training(
                target_cols=[target],
                input_cols=cfg.get("input", []),
                past_cols=cfg.get("past", []),
                run_tag=f"target_{target}",
                artifact_group="per_target",
                use_detrend=use_detrend,
                use_log_transform=use_log_transform,
                use_box_cox=use_box_cox,
                shuffle_data=shuffle_data,
                exp_smoothing_alpha=args.exp_smoothing_alpha,
                eval_on_smoothed=args.eval_on_smoothed,
                model_names=model_names,
                exclude_model_names=exclude_model_names,
                model_groups=model_groups,
                max_files_to_load=max_files_to_load,
                use_builtin_scalers=use_builtin_scalers,
                split_per_session=args.split_per_session,
                disable_mape=disable_mape,
                clip_predictions=clip_predictions,
            )

    if "simple" in runs:
        simple_target_cols = SIMPLE_MODEL_CONFIG.get("target_cols", TARGET_COLS)
        simple_input_cols = SIMPLE_MODEL_CONFIG.get("input_cols", SIMPLE_MODEL_CONFIG.get("input", []))
        simple_past_cols = SIMPLE_MODEL_CONFIG.get("past_cols", SIMPLE_MODEL_CONFIG.get("past", []))
        run_training(
            target_cols=simple_target_cols,
            input_cols=simple_input_cols,
            past_cols=simple_past_cols,
            run_tag="simple_model",
            artifact_group="simple",
            use_detrend=use_detrend,
            use_log_transform=use_log_transform,
            use_box_cox=use_box_cox,
            shuffle_data=shuffle_data,
            exp_smoothing_alpha=args.exp_smoothing_alpha,
            eval_on_smoothed=args.eval_on_smoothed,
            model_names=model_names,
            exclude_model_names=exclude_model_names,
            model_groups=model_groups,
            max_files_to_load=max_files_to_load,
            use_builtin_scalers=use_builtin_scalers,
            split_per_session=args.split_per_session,
            disable_mape=disable_mape,
            clip_predictions=clip_predictions,
        )