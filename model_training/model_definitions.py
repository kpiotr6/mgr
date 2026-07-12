"""
Model definitions for time series forecasting.
Centralizes model configuration and instantiation logic.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch

from darts import TimeSeries
from darts.models import (
    LinearRegressionModel,
    NeuralForecastModel,
    DLinearModel,
    NLinearModel,
    XGBModel,
)

try:
    # Darts >= 0.44
    from darts.models import Chronos2Model  # type: ignore

    _HAS_CHRONOS2 = True
except Exception:
    Chronos2Model = None  # type: ignore
    _HAS_CHRONOS2 = False
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import os

from neuralforecast.losses.pytorch import RMSE, MSE

# ---------------------------------------------------------
# Monkey-patch neuralforecast BaseModel to support ReduceLROnPlateau
# without raising PyTorch Lightning's MisconfigurationException.
# ---------------------------------------------------------
try:
    import neuralforecast.common._base_model as nf_base_model
    _original_nf_configure_optimizers = nf_base_model.BaseModel.configure_optimizers

    def _patched_configure_optimizers(self):
        res = _original_nf_configure_optimizers(self)
        if isinstance(res, dict) and "lr_scheduler" in res:
            sch = res["lr_scheduler"]
            if isinstance(sch, dict):
                scheduler_obj = sch.get("scheduler")
                if isinstance(scheduler_obj, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if "monitor" not in sch:
                        sch["monitor"] = "ptl/val_loss"
            else:
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    res["lr_scheduler"] = {
                        "scheduler": sch,
                        "monitor": "ptl/val_loss"
                    }
        return res

    nf_base_model.BaseModel.configure_optimizers = _patched_configure_optimizers
except Exception:
    pass
# ---------------------------------------------------------


NAIVE_MODELS = ["NaiveLastValue"]
MODELS_WITH_FUTURE_COVARIATES = ["LinearRegression", "Lasso", "NeuralForecast_Nhits", "NeuralForecast_NLinear", "NeuralForecast_DLinear", "NeuralForecast_TFT", "NeuralForecast_TSMixer", "NeuralForecast_PatchTST", "NeuralForecast_XLinear", "NeuralForecast_xLSTM", "NeuralForecast_RNN", "NeuralForecast_BiTCN", "Darts_XGB", "Darts_DLinear", "Darts_NLinear", "Darts_Nhits", "Darts_TSMixer", "Chronos2"]
MODELS_FUTURE_COVARIATES_ONLY = ["NeuralForecast_TimesNet"]
CHECKPOINT_MODELS = ["NeuralForecast_Nhits", "NeuralForecast_NLinear", "NeuralForecast_DLinear", "NeuralForecast_TFT", "NeuralForecast_TSMixer", "NeuralForecast_PatchTST", "NeuralForecast_XLinear", "NeuralForecast_xLSTM", "NeuralForecast_TimesNet", "NeuralForecast_RNN", "NeuralForecast_BiTCN", "Darts_DLinear", "Darts_NLinear", "Darts_Nhits", "Darts_TSMixer", "Darts_RNN"]


class LastValueRepeater:
    def __init__(self):
        self._series = None

    def fit(self, series, *args, **kwargs):
        self._series = series[-1] if isinstance(series, list) else series
        return self

    def predict(self, n, series=None, **kwargs):
        source_series = series[-1] if isinstance(series, list) else series
        if source_series is None:
            source_series = self._series
        if source_series is None:
            raise ValueError("LastValueRepeater requires a fitted or provided series")

        last_values = np.asarray(source_series.last_values())
        forecast_values = np.repeat(last_values[np.newaxis, :], n, axis=0)

        if source_series.has_datetime_index:
            future_times = pd.date_range(
                start=source_series.end_time() + source_series.freq,
                periods=n,
                freq=source_series.freq,
            )
        else:
            step = source_series.freq if getattr(source_series, "freq", None) is not None else 1
            start = source_series.end_time() + step
            stop = start + step * n
            future_times = pd.RangeIndex(start=start, stop=stop, step=step)

        return TimeSeries.from_times_and_values(
            future_times,
            forecast_values,
            columns=source_series.components,
        )

class LossPlotCallback(Callback):
    def __init__(self, model_name, save_dir):
        super().__init__()
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.plot_path = self.save_dir / f"{self.model_name}_loss.png"
        self.train_losses = []
        self.val_losses = []
        os.makedirs(self.save_dir, exist_ok=True)

    def _extract_metric(self, metrics, exact_names, prefixes):
        for key in exact_names:
            if key in metrics:
                value = metrics[key]
                try:
                    return float(value.item())
                except Exception:
                    try:
                        return float(value)
                    except Exception:
                        continue

        for key, value in metrics.items():
            lower_key = key.lower()
            if any(prefix in lower_key for prefix in prefixes):
                try:
                    return float(value.item())
                except Exception:
                    try:
                        return float(value)
                    except Exception:
                        continue
        return None

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = self._extract_metric(
            metrics,
            ("train_loss", "train_loss_epoch", "train_loss_step"),
            ("train_loss", "train/"),
        )
        if train_loss is not None:
            self.train_losses.append(train_loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = self._extract_metric(
            metrics,
            ("val_loss", "val_loss_epoch", "valid_loss", "valid_loss_epoch"),
            ("val_loss", "valid_loss", "val/"),
        )
        if val_loss is not None:
            self.val_losses.append(val_loss)
            self.plot_losses()

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        if self.train_losses:
            train_epochs = list(range(1, len(self.train_losses) + 1))
            plt.plot(train_epochs, self.train_losses, marker='o', label='Training Loss')

        if self.val_losses:
            val_epochs = list(range(1, len(self.val_losses) + 1))
            plt.plot(val_epochs, self.val_losses, marker='o', label='Validation Loss')

        for epoch, value in enumerate(self.train_losses, start=1):
            plt.text(epoch, value, f'{value:.4f}', fontsize=9, ha='right', va='bottom')

        for epoch, value in enumerate(self.val_losses, start=1):
            plt.text(epoch, value, f'{value:.4f}', fontsize=9, ha='left', va='bottom')

        plt.title(f'Training and Validation Loss - {self.model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()


def _build_run_artifact_root(input_chunk_length: int, output_chunk_length: int, run_group: str) -> Path:
    return Path("outputs/logs") / f"I{input_chunk_length}_O{output_chunk_length}" / run_group


def _build_nf_trainer_kwargs(
    artifact_root: Path,
    model_log_name: str,
) -> dict:
    model_root = artifact_root / model_log_name
    return {
        "logger": CSVLogger(str(artifact_root), name=model_log_name),
        "callbacks": [LossPlotCallback(model_log_name, model_root / "plots")],
        "log_every_n_steps": 1,
        "enable_model_summary": False,
    }

def get_models(
    input_chunk_length: int,
    output_chunk_length: int,
    input_cols: list[str] | None = None,
    past_cols: list[str] | None = None,
    run_group: str = "all_targets",
) -> dict:
    """
    Create and return a dictionary of models with specified input/output chunk lengths.
    """
    input_cols = input_cols or []
    past_cols = past_cols or []

    has_input_covariates = len(input_cols) > 0
    has_past_covariates = len(past_cols) > 0
    artifact_root = _build_run_artifact_root(input_chunk_length, output_chunk_length, run_group)

    linear_regression_kwargs = {
        "lags": input_chunk_length,
        "output_chunk_length": output_chunk_length,
    }
    if has_past_covariates:
        linear_regression_kwargs["lags_past_covariates"] = input_chunk_length
    if has_input_covariates:
        linear_regression_kwargs["lags_future_covariates"] = (input_chunk_length, output_chunk_length)

    models = {
        "NaiveLastValue": LastValueRepeater(),
        "LinearRegression": LinearRegressionModel(**linear_regression_kwargs),
        "NeuralForecast_BiTCN": NeuralForecastModel(
            model_name=f"NeuralForecast_BiTCN_I{input_chunk_length}_O{output_chunk_length}",
            model="BiTCN",
            save_checkpoints=True,
            force_reset=True,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            loss_fn=RMSE(),
            n_epochs=20,
            batch_size=128,
            optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-4},
            model_kwargs={
                "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "lr_scheduler_kwargs": {"mode": "min", "factor": 0.5, "patience": 2},
            },
            pl_trainer_kwargs=_build_nf_trainer_kwargs(artifact_root, "NF_BITCN"),
        ),
        "NeuralForecast_TSMixer": NeuralForecastModel(
            model_name=f"NeuralForecast_TSMixer_I{input_chunk_length}_O{output_chunk_length}",
            model="TSMixerx",
            save_checkpoints=True,
            force_reset=True,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            model_kwargs={
                "n_block": 8,
                "ff_dim": 128,
                "dropout": 0.1,
                "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "lr_scheduler_kwargs": {"mode": "min", "factor": 0.5, "patience": 10},
            },
            loss_fn=RMSE(),
            n_epochs=40,
            batch_size=1024,
            optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-4},
            pl_trainer_kwargs=_build_nf_trainer_kwargs(artifact_root, "NF_TSMIXER"),
        ),
        "NeuralForecast_Nhits": NeuralForecastModel(
            model_name=f"NeuralForecast_Nhits_I{input_chunk_length}_O{output_chunk_length}",
            model="NHITS",
            save_checkpoints=True,
            force_reset=True,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            loss_fn=RMSE(),
            n_epochs=20,
            batch_size=1024,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-4},
            model_kwargs={
                "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "lr_scheduler_kwargs": {"mode": "min", "factor": 0.5, "patience": 2},
                "mlp_units": [[1024, 1024, 1024], [1024, 1024, 1024], [1024, 1024, 1024]],
                "n_blocks": [2, 2, 2], # Increased blocks per stack
                "n_pool_kernel_size": [8, 4, 1], # Can be adjusted based on input sequence length
                "n_freq_downsample": [8, 4, 1],
                "activation": "LeakyReLU"
            },
            pl_trainer_kwargs=_build_nf_trainer_kwargs(artifact_root, "NF_NHITS"),
        ),
        "NeuralForecast_XLinear": NeuralForecastModel(
            model_name=f"NeuralForecast_XLinear_I{input_chunk_length}_O{output_chunk_length}",
            model="XLinear",
            save_checkpoints=True,
            force_reset=True,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            loss_fn=RMSE(),
            n_epochs=200,
            batch_size=2048,
            optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-4},
            model_kwargs={
                "hidden_size": 512,
                "temporal_ff": 512,
                "channel_ff": 256,
                "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "lr_scheduler_kwargs": {"mode": "min", "factor": 0.5, "patience": 2},
            },
            pl_trainer_kwargs=_build_nf_trainer_kwargs(artifact_root, "NF_XLINEAR"),
        ),
        "NeuralForecast_TFT": NeuralForecastModel(
            model_name=f"NeuralForecast_TFT_I{input_chunk_length}_O{output_chunk_length}",
            model="TFT",
            save_checkpoints=True,
            force_reset=True,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            loss_fn=RMSE(),
            n_epochs=20,
            batch_size=128,
            optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-4},
            model_kwargs={

                "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "lr_scheduler_kwargs": {"mode": "min", "factor": 0.5, "patience": 2},
            },
            pl_trainer_kwargs=_build_nf_trainer_kwargs(artifact_root, "NF_TFT"),
        ),
    }

    return models