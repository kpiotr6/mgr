"""
Model definitions for time series forecasting.
Centralizes model configuration and instantiation logic.
"""

from darts.models import (
    NaiveSeasonal,
    LinearRegressionModel,
    NeuralForecastModel,
)
from sklearn.linear_model import Lasso
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import os
from config import INPUT_COLS, PAST_COLS
from neuralforecast.losses.pytorch import MAPE

NAIVE_MODELS = ["NaiveLastValue"]
MODELS_WITH_FUTURE_COVARIATES = ["LinearRegression", "Lasso", "NeuralForecast_Nhits", "NeuralForecast_NLinear", "NeuralForecast_DLinear", "NeuralForecast_TFT", "NeuralForecast_TSMixer", "Darts_Nhits", "Darts_TSMixer"]
MODELS_PAST_COVARIATES_ONLY = []
CHECKPOINT_MODELS = ["NeuralForecast_Nhits", "NeuralForecast_NLinear", "NeuralForecast_DLinear", "NeuralForecast_TFT", "NeuralForecast_TSMixer", "Darts_Nhits", "Darts_TSMixer"]

class LossPlotCallback(Callback):
    def __init__(self, model_name, save_dir):
        super().__init__()
        self.model_name = model_name
        self.save_dir = save_dir
        self.val_losses = []
        os.makedirs(self.save_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        # PyTorch Lightning validation loss is typically keyed with 'val_loss' or 'valid_loss'
        # Darts/NeuralForecast generally logs 'val_loss'
        for key in metrics:
            if 'val_loss' in key.lower():
                val_loss = metrics[key].item()
                self.val_losses.append(val_loss)
                self.plot_losses()
                break

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        epochs = list(range(1, len(self.val_losses) + 1))
        plt.plot(epochs, self.val_losses, marker='o', label='Validation Loss')

        for i, val in enumerate(self.val_losses):
            plt.text(epochs[i], val, f'{val:.4f}', fontsize=9, ha='right', va='bottom')

        plt.title(f'Validation Loss during Training - {self.model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.model_name}_val_loss.png'))
        plt.close()

def get_models(input_chunk_length: int, output_chunk_length: int) -> dict:
    """
    Create and return a dictionary of models with specified input/output chunk lengths.

    Parameters
    ----------
    input_chunk_length : int
        The input chunk length for the models
    output_chunk_length : int
        The output chunk length for the models

    Returns
    -------
    dict
        Dictionary mapping model names to model instances
    """
    has_input_covariates = len(INPUT_COLS) > 0
    has_past_covariates = len(PAST_COLS) > 0

    linear_regression_kwargs = {
        "lags": input_chunk_length,
        "output_chunk_length": output_chunk_length,
    }
    if has_past_covariates:
        linear_regression_kwargs["lags_past_covariates"] = input_chunk_length
    if has_input_covariates:
        linear_regression_kwargs["lags_future_covariates"] = (input_chunk_length, output_chunk_length)

    models = {
        "NaiveLastValue": NaiveSeasonal(K=1),
        "LinearRegression": LinearRegressionModel(**linear_regression_kwargs),
        "NeuralForecast_TSMixer": NeuralForecastModel(
            model_name=f"NeuralForecast_TSMixer_I{input_chunk_length}_O{output_chunk_length}",
            model="TSMixerx",
            save_checkpoints=True,
            force_reset=True,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            loss_fn=MAPE(),
            n_epochs=5,
            batch_size=128,
            optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-4},
            pl_trainer_kwargs={
                "logger": CSVLogger(f"outputs/logs/I{input_chunk_length}_O{output_chunk_length}", name="NF_TSMIXER"),
                "callbacks": [LossPlotCallback("NF_TSMIXER", f"outputs/logs/I{input_chunk_length}_O{output_chunk_length}/plots")],
                "log_every_n_steps": 1,
                "enable_model_summary": False,
            }
        ),
        "NeuralForecast_Nhits": NeuralForecastModel(
            model_name=f"NeuralForecast_Nhits_I{input_chunk_length}_O{output_chunk_length}",
            model="NHITS",
            save_checkpoints=True,
            force_reset=True,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            loss_fn=MAPE(),
            n_epochs=5,
            batch_size=128,
            optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-4},
            pl_trainer_kwargs={
                "logger": CSVLogger(f"outputs/logs/I{input_chunk_length}_O{output_chunk_length}", name="NF_NHITS"),
                "callbacks": [LossPlotCallback("NF_NHITS", f"outputs/logs/I{input_chunk_length}_O{output_chunk_length}/plots")],
                "log_every_n_steps": 1,
                "enable_model_summary": False,
            }
        ),
    }

    return models
