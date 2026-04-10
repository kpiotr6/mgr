import pandas as pd
from darts import TimeSeries
from darts.models import TFTModel, BlockRNNModel, NaiveSeasonal, NLinearModel, DLinearModel, XGBModel, RandomForest, TSMixerModel, NHiTSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mse
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
from config import INPUT_COLS, TARGET_COLS, TIME_COL, DEFAULT_FREQ
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
import os
import glob
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau


def load_data(filepath: str):
    df = pd.read_csv(filepath)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    # A new time series starts when 'is_at_edge' switches from False to True
    # If the file starts with False, it belongs to the first series (group 0)
    group_id = (df['is_at_edge'].astype(int).diff() == 1).cumsum()

    targets = []
    covariates = []

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
        cov_ts = TimeSeries.from_dataframe(
            group_df,
            time_col=TIME_COL,
            value_cols=INPUT_COLS,
        )
        covariates.append(cov_ts)

    return targets, covariates

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    targets_list = []
    covariates_list = []

    # Number of files to use for dataset creation (None to use all files)
    MAX_FILES_TO_LOAD = 1

    data_dir = "data_preprocessed"
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))

    all_files = sorted(all_files)
    if MAX_FILES_TO_LOAD is not None:
        all_files = all_files[:MAX_FILES_TO_LOAD]

    for filepath in all_files:
        t, c = load_data(filepath)
        targets_list.extend(t)
        covariates_list.extend(c)
        print(f"Loaded {len(t)} series blocks from {filepath}")

    print(f"Total loaded series blocks: {len(targets_list)}")

    # Shuffle the blocks to randomize train/test datasets
    random.seed(42)
    combined = list(zip(targets_list, covariates_list))
    random.shuffle(combined)
    targets_list, covariates_list = zip(*combined)

    # Needs to be a list again
    targets_list = list(targets_list)
    covariates_list = list(covariates_list)

    # Split into train, val, test (70% / 15% / 15%)
    n_total = len(targets_list)
    split_idx_1 = int(n_total * 0.7)
    split_idx_2 = int(n_total * 0.85)

    train_targets = targets_list[:split_idx_1]
    train_covariates = covariates_list[:split_idx_1]

    val_targets = targets_list[split_idx_1:split_idx_2]
    val_covariates = covariates_list[split_idx_1:split_idx_2]

    test_targets = targets_list[split_idx_2:]
    test_covariates = covariates_list[split_idx_2:]

    print(f"Train blocks: {len(train_targets)}, Val blocks: {len(val_targets)}, Test blocks: {len(test_targets)}")

    # Scale the data
    target_scaler = Scaler(global_fit=True)
    covariates_scaler = Scaler(global_fit=True)

    train_targets_scaled = target_scaler.fit_transform(train_targets)
    train_covariates_scaled = covariates_scaler.fit_transform(train_covariates)

    val_targets_scaled = target_scaler.transform(val_targets)
    val_covariates_scaled = covariates_scaler.transform(val_covariates)

    test_targets_scaled = target_scaler.transform(test_targets)
    test_covariates_scaled = covariates_scaler.transform(test_covariates)

    INPUT_CHUNK_LENGTH = 90
    OUTPUT_CHUNK_LENGTH = 90

    # Metrics for PyTorch Lightning logging
    metrics = MetricCollection([MeanAbsoluteError(), MeanSquaredError()])

    # Models definition
    models = {
        "NaiveLastValue": NaiveSeasonal(K=1),
        # "BlockRNN": BlockRNNModel(
        #     model_name="BlockRNN",
        #     save_checkpoints=True,
        #     force_reset=True,
        #     model="GRU",
        #     hidden_dim=64,
        #     n_rnn_layers=2,
        #     input_chunk_length=INPUT_CHUNK_LENGTH,
        #     output_chunk_length=OUTPUT_CHUNK_LENGTH,
        #     n_epochs=20,
        #     batch_size=32,
        #     optimizer_kwargs={"lr": 1e-3},
        #     lr_scheduler_cls=ReduceLROnPlateau,
        #     lr_scheduler_kwargs={"factor": 0.5, "patience": 3, "monitor": "val_loss"},
        #     pl_trainer_kwargs={"logger": CSVLogger("outputs/logs", name="BlockRNN"), "log_every_n_steps": 1}
        # ),

        "NLinear": NLinearModel(
            model_name="NLinear",
            save_checkpoints=True,
            force_reset=True,
            input_chunk_length=INPUT_CHUNK_LENGTH,
            output_chunk_length=OUTPUT_CHUNK_LENGTH,
            const_init=False,
            n_epochs=40,
            batch_size=128,
            optimizer_kwargs={"lr": 1e-3},
            torch_metrics=metrics.clone(),
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs={"factor": 0.5, "patience": 5, "monitor": "val_loss"},
            pl_trainer_kwargs={
                "logger": CSVLogger("outputs/logs", name="NLinear"),
                "log_every_n_steps": 1
            }
        ),
        "DLinear": DLinearModel(
            model_name="DLinear",
            save_checkpoints=True,
            force_reset=True,
            input_chunk_length=INPUT_CHUNK_LENGTH,
            output_chunk_length=OUTPUT_CHUNK_LENGTH,
            const_init=False,
            n_epochs=40,
            batch_size=128,
            optimizer_kwargs={"lr": 1e-3},
            torch_metrics=metrics.clone(),
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs={"factor": 0.5, "patience": 5, "monitor": "val_loss"},
            pl_trainer_kwargs={
                "logger": CSVLogger("outputs/logs", name="DLinear"),
                "log_every_n_steps": 1
            }
        ),
        # "TFT": TFTModel(
        #     model_name="TFT",
        #     save_checkpoints=True,
        #     force_reset=True,
        #     input_chunk_length=INPUT_CHUNK_LENGTH,
        #     output_chunk_length=OUTPUT_CHUNK_LENGTH,
        #     hidden_size=128,
        #     lstm_layers=2,
        #     num_attention_heads=4,
        #     dropout=0.1,
        #     add_relative_index=True,
        #     n_epochs=20,
        #     batch_size=32,
        #     optimizer_kwargs={"lr": 1e-3},
        #     lr_scheduler_cls=ReduceLROnPlateau,
        #     lr_scheduler_kwargs={"factor": 0.5, "patience": 3, "monitor": "val_loss"},
        #     pl_trainer_kwargs={
        #         "logger": CSVLogger("outputs/logs", name="TFT"),
        #         "log_every_n_steps": 1
        #     }
        # ),
        # "TSMixer": TSMixerModel(
        #     model_name="TSMixer",
        #     save_checkpoints=True,
        #     force_reset=True,
        #     input_chunk_length=INPUT_CHUNK_LENGTH,
        #     output_chunk_length=OUTPUT_CHUNK_LENGTH,
        #     hidden_size=64,
        #     ff_size=64,
        #     num_blocks=3,
        #     dropout=0.1,
        #     n_epochs=20,
        #     batch_size=32,
        #     optimizer_kwargs={"lr": 1e-3},
        #     lr_scheduler_cls=ReduceLROnPlateau,
        #     lr_scheduler_kwargs={"factor": 0.5, "patience": 3, "monitor": "val_loss"},
        #     pl_trainer_kwargs={
        #         "logger": CSVLogger("outputs/logs", name="TSMixer"),
        #         "log_every_n_steps": 1
        #     }
        # ),
        # "NHiTS": NHiTSModel(
        #     model_name="NHiTS",
        #     save_checkpoints=True,
        #     force_reset=True,
        #     input_chunk_length=INPUT_CHUNK_LENGTH,
        #     output_chunk_length=OUTPUT_CHUNK_LENGTH,
        #     n_epochs=20,
        #     batch_size=32,
        #     optimizer_kwargs={"lr": 1e-3},
        #     lr_scheduler_cls=ReduceLROnPlateau,
        #     lr_scheduler_kwargs={"factor": 0.5, "patience": 4, "monitor": "val_loss"},
        #     pl_trainer_kwargs={
        #         "logger": CSVLogger("outputs/logs", name="NHiTS"),
        #         "log_every_n_steps": 1
        #     }
        # ),
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Naive models don't use covariates, and don't need training on a whole list
        if name == "NaiveLastValue":
            # For naive, we just evaluate on each test block
            pass # No training required
        elif name in ["TFT", "NLinear", "DLinear", "XGBoost", "RandomForest", "TSMixer"]:
            model.fit(
                series=train_targets_scaled,
                future_covariates=train_covariates_scaled,
                val_series=val_targets_scaled,
                val_future_covariates=val_covariates_scaled
            )
        elif name in ["BlockRNN", "NHiTS"]:
            # BlockRNN and NHiTS accept covariates via the past_covariates argument
            model.fit(
                series=train_targets_scaled,
                past_covariates=train_covariates_scaled,
                val_series=val_targets_scaled,
                val_past_covariates=val_covariates_scaled
            )
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

        if name in ["TFT", "NLinear", "DLinear", "BlockRNN", "TSMixer", "NHiTS"]:
            try:
                # Load the best model from checkpoint for testing
                model = type(model).load_from_checkpoint(model_name=name, best=True)
                print(f"Loaded best checkpoint for {name}.")
            except Exception as e:
                print(f"Could not load best checkpoint for {name}: {e}")

        # Calculate training MAE/MSE (optional, can be very slow for large datasets)
        if name not in ["NaiveLastValue"]:
            try:
                # To keep it quick, we'll just evaluate on a subset or full train
                train_preds_scaled = model.predict(n=OUTPUT_CHUNK_LENGTH, series=[t[:-OUTPUT_CHUNK_LENGTH] for t in train_targets_scaled if len(t) > INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH],
                                            future_covariates=[c for t, c in zip(train_targets_scaled, train_covariates_scaled) if len(t) > INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH] if name in ["TFT", "NLinear", "DLinear", "XGBoost", "RandomForest", "TSMixer"] else None,
                                            past_covariates=[c for t, c in zip(train_targets_scaled, train_covariates_scaled) if len(t) > INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH] if name in ["BlockRNN", "NHiTS"] else None)

                # Inverse transform the predictions back to original distribution
                if isinstance(train_preds_scaled, list):
                    train_preds = target_scaler.inverse_transform(train_preds_scaled)
                else:
                    train_preds = target_scaler.inverse_transform([train_preds_scaled])

                true_train = [t[-OUTPUT_CHUNK_LENGTH:] for t in train_targets if len(t) > INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH]
                train_mae = mae(true_train, train_preds)
                train_mse = mse(true_train, train_preds)
                print(f"Training metrics for {name} - MAE: {train_mae:.4f}, MSE: {train_mse:.4f}")
            except Exception as e:
                print(f"Could not calculate train metrics for {name}: {e}")

        mae_list = []
        mse_list = []
        mae_cols = {col: [] for col in TARGET_COLS}
        mse_cols = {col: [] for col in TARGET_COLS}
        all_preds = []
        all_trues = []

        for i, (ts_target, ts_cov, ts_target_raw) in enumerate(zip(test_targets_scaled, test_covariates_scaled, test_targets)):
            # Predict from the last OUTPUT_CHUNK_LENGTH steps
            # Ensure the series is long enough
            if len(ts_target) <= INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH:
                continue

            y_train = ts_target[:-OUTPUT_CHUNK_LENGTH]

            # y_true needs to be unscaled for metric calculation later
            y_true = ts_target_raw[-OUTPUT_CHUNK_LENGTH:]

            if name in ["TFT", "NLinear", "DLinear", "XGBoost", "RandomForest", "TSMixer"]:
                pred_scaled = model.predict(n=OUTPUT_CHUNK_LENGTH, series=y_train, future_covariates=ts_cov)
            elif name in ["BlockRNN", "NHiTS"]:
                pred_scaled = model.predict(n=OUTPUT_CHUNK_LENGTH, series=y_train, past_covariates=ts_cov)
            elif name == "NaiveLastValue":
                model.fit(y_train)
                pred_scaled = model.predict(n=OUTPUT_CHUNK_LENGTH)
            else:
                pred_scaled = model.predict(n=OUTPUT_CHUNK_LENGTH, series=y_train)

            # Inverse scale predictions directly mapping back real boundaries
            pred = target_scaler.inverse_transform(pred_scaled)

            mae_list.append(mae(y_true, pred))
            mse_list.append(mse(y_true, pred))

            for col in TARGET_COLS:
                mae_cols[col].append(mae(y_true[col], pred[col]))
                mse_cols[col].append(mse(y_true[col], pred[col]))

            pred_df = pred.to_dataframe()
            pred_df['block_idx'] = i
            all_preds.append(pred_df)

            # Save true values only on the first model pass
            if name == list(models.keys())[0]:
                true_df = y_true.to_dataframe()
                true_df['block_idx'] = i
                all_trues.append(true_df)

        if len(mae_list) > 0:
            avg_mae = sum(mae_list) / len(mae_list)
            avg_mse = sum(mse_list) / len(mse_list)

            res_dict = {
                "Model": name,
                "MAE": avg_mae,
                "MSE": avg_mse
            }

            print(f"Results for {name}: MAE={avg_mae:.4f}, MSE={avg_mse:.4f}")
            for col in TARGET_COLS:
                avg_mae_col = sum(mae_cols[col]) / len(mae_cols[col])
                avg_mse_col = sum(mse_cols[col]) / len(mse_cols[col])
                res_dict[f"MAE_{col}"] = avg_mae_col
                res_dict[f"MSE_{col}"] = avg_mse_col
                print(f"  {col} - MAE: {avg_mae_col:.4f}, MSE: {avg_mse_col:.4f}")

            results.append(res_dict)

            # Save to CSV
            pd.concat(all_preds).to_csv(f"outputs/pred_{name}.csv")
            if all_trues:
                pd.concat(all_trues).to_csv("outputs/true_values.csv")
        else:
            print(f"No valid test series for {name} (too short)")

    print("\nGenerating training loss chart...")
    plt.figure(figsize=(10, 6))
    for name in models.keys():
        if name in ["NaiveLastValue", "XGBoost", "RandomForest"]:
            continue
        try:
            metrics_paths = glob.glob(f"outputs/logs/{name}/*/metrics.csv")
            if metrics_paths:
                # Sort by creation time to get the latest
                metrics_paths.sort(key=os.path.getmtime)
                metrics_df = pd.read_csv(metrics_paths[-1])

                # Check for train_loss_epoch first (whole epoch loss)
                if 'train_loss_epoch' in metrics_df.columns:
                    epoch_loss = metrics_df['train_loss_epoch'].dropna()
                    # Some loggers append NaNs for step logs, so reset index
                    plt.plot(range(len(epoch_loss)), epoch_loss.values, label=f"{name} (epoch)")
                elif 'train_loss' in metrics_df.columns and 'epoch' in metrics_df.columns:
                    # Fallback: Group by epoch and calculate the mean training loss
                    epoch_loss = metrics_df.groupby('epoch')['train_loss'].mean().dropna()
                    plt.plot(epoch_loss.index, epoch_loss.values, label=f"{name} (avg/epoch)")
                elif 'train_loss' in metrics_df.columns:
                    loss_series = metrics_df['train_loss'].dropna()
                    plt.plot(loss_series.values, label=f"{name} (steps)")

        except Exception as e:
            print(f"Could not load/plot loss for {name}: {e}")

    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    chart_path = "outputs/training_loss.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"Training loss chart saved to {chart_path}")

    print("\nGenerating MAE and MSE charts...")
    for metric_name in ['MeanAbsoluteError', 'MeanSquaredError']:
        for prefix, phase in zip(['train_', 'val_'], ['Training', 'Validation']):
            plt.figure(figsize=(10, 6))
            col_name = f'{prefix}{metric_name}'
            col_name_epoch = f'{prefix}{metric_name}_epoch'

            plotted = False
            for name in models.keys():
                if name in ["NaiveLastValue", "XGBoost", "RandomForest"]:
                    continue
                try:
                    metrics_paths = glob.glob(f"outputs/logs/{name}/*/metrics.csv")
                    if metrics_paths:
                        metrics_paths.sort(key=os.path.getmtime)
                        metrics_df = pd.read_csv(metrics_paths[-1])

                        if col_name_epoch in metrics_df.columns:
                            epoch_metric = metrics_df[col_name_epoch].dropna()
                            plt.plot(range(len(epoch_metric)), epoch_metric.values, label=f"{name}")
                            plotted = True
                        elif col_name in metrics_df.columns and 'epoch' in metrics_df.columns:
                            epoch_metric = metrics_df.groupby('epoch')[col_name].mean().dropna()
                            plt.plot(epoch_metric.index, epoch_metric.values, label=f"{name}")
                            plotted = True
                except Exception as e:
                    pass

            if plotted:
                plt.xlabel('Epoch')
                plt.ylabel(metric_name)
                plt.title(f'{phase} {metric_name}')
                plt.legend()
                plt.grid(True)
                chart_path_metric = f'outputs/{phase.lower()}_{metric_name.lower()}.png'
                plt.savefig(chart_path_metric)
                print(f"{phase} {metric_name} chart saved to {chart_path_metric}")
            plt.close()

    if results:
        results_df = pd.DataFrame(results)
        results_path = "outputs/evaluation_metrics.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nFinal evaluation metrics saved to {results_path}")

