import pandas as pd
from darts import TimeSeries
from darts.models import TFTModel, BlockRNNModel, NaiveSeasonal, NLinearModel
from darts.metrics import mae, mse
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
from config import INPUT_COLS, TARGET_COLS, TIME_COL, DEFAULT_FREQ
import torch
import os
import glob


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

        if len(group_df) < 540:
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

    data_dir = "data_preprocessed"
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))

    for filepath in sorted(all_files):
        t, c = load_data(filepath)
        targets_list.extend(t)
        covariates_list.extend(c)
        print(f"Loaded {len(t)} series blocks from {filepath}")

    print(f"Total loaded series blocks: {len(targets_list)}")

    # Split into train and test
    split_idx = int(len(targets_list) * 0.8)
    train_targets = targets_list[:split_idx]
    train_covariates = covariates_list[:split_idx]
    test_targets = targets_list[split_idx:]
    test_covariates = covariates_list[split_idx:]

    print(f"Train blocks: {len(train_targets)}, Test blocks: {len(test_targets)}")

    INPUT_CHUNK_LENGTH = 360
    OUTPUT_CHUNK_LENGTH = 180

    # Models definition
    models = {
        "NaiveLastValue": NaiveSeasonal(K=1),
        "BlockRNN": BlockRNNModel(
            model="GRU",
            input_chunk_length=INPUT_CHUNK_LENGTH,
            output_chunk_length=OUTPUT_CHUNK_LENGTH,
            n_epochs=5,
            pl_trainer_kwargs={"logger": CSVLogger("outputs/logs", name="BlockRNN")}
        ),
        "TFT": TFTModel(
            input_chunk_length=INPUT_CHUNK_LENGTH,
            output_chunk_length=OUTPUT_CHUNK_LENGTH,
            add_relative_index=True,
            n_epochs=5,
            pl_trainer_kwargs={"logger": CSVLogger("outputs/logs", name="TFT")}
        ),
        "NLinear": NLinearModel(
            input_chunk_length=INPUT_CHUNK_LENGTH,
            output_chunk_length=OUTPUT_CHUNK_LENGTH,
            n_epochs=5,
            pl_trainer_kwargs={"logger": CSVLogger("outputs/logs", name="NLinear")}
        )
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Naive models don't use covariates, and don't need training on a whole list
        if name == "NaiveLastValue":
            # For naive, we just evaluate on each test block
            pass # No training required
        elif name in ["TFT", "NLinear"]:
            model.fit(series=train_targets, future_covariates=train_covariates)
        elif name == "BlockRNN":
            # BlockRNN accepts covariates via the past_covariates argument
            model.fit(series=train_targets, past_covariates=train_covariates)
        else:
            try:
                model.fit(series=train_targets)
            except Exception as e:
                print(f"Error training {name}: {e}")

        # Testing
        print(f"Testing {name}...")
        mae_list = []
        mse_list = []
        all_preds = []
        all_trues = []

        for i, (ts_target, ts_cov) in enumerate(zip(test_targets, test_covariates)):
            # Predict from the last OUTPUT_CHUNK_LENGTH steps
            # Ensure the series is long enough
            if len(ts_target) <= INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH:
                continue

            y_train = ts_target[:-OUTPUT_CHUNK_LENGTH]
            y_true = ts_target[-OUTPUT_CHUNK_LENGTH:]

            if name in ["TFT", "NLinear"]:
                pred = model.predict(n=OUTPUT_CHUNK_LENGTH, series=y_train, future_covariates=ts_cov)
            elif name == "BlockRNN":
                pred = model.predict(n=OUTPUT_CHUNK_LENGTH, series=y_train, past_covariates=ts_cov)
            elif name == "NaiveLastValue":
                model.fit(y_train)
                pred = model.predict(n=OUTPUT_CHUNK_LENGTH)
            else:
                pred = model.predict(n=OUTPUT_CHUNK_LENGTH, series=y_train)

            mae_list.append(mae(y_true, pred))
            mse_list.append(mse(y_true, pred))

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
            print(f"Results for {name}: MAE={avg_mae:.4f}, MSE={avg_mse:.4f}")

            # Save to CSV
            pd.concat(all_preds).to_csv(f"outputs/pred_{name}.csv")
            if all_trues:
                pd.concat(all_trues).to_csv("outputs/true_values.csv")
        else:
            print(f"No valid test series for {name} (too short)")

    print("\nGenerating training loss chart...")
    plt.figure(figsize=(10, 6))
    for name in models.keys():
        if name == "NaiveLastValue":
            continue
        try:
            metrics_paths = glob.glob(f"outputs/logs/{name}/*/metrics.csv")
            if metrics_paths:
                # Sort by creation time to get the latest
                metrics_paths.sort(key=os.path.getmtime)
                metrics_df = pd.read_csv(metrics_paths[-1])
                if 'train_loss' in metrics_df.columns:
                    loss_series = metrics_df['train_loss'].dropna()
                    plt.plot(loss_series.values, label=f"{name}")
        except Exception as e:
            print(f"Could not load/plot loss for {name}: {e}")

    plt.title("Training Loss per Epoch/Step")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    chart_path = "outputs/training_loss.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"Training loss chart saved to {chart_path}")

