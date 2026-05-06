"""
Model definitions for time series forecasting.
Centralizes model configuration and instantiation logic.
"""

from darts.models import (
    NaiveSeasonal,
    LinearRegressionModel,
    NeuralForecastModel,
)
from pytorch_lightning.loggers import CSVLogger
from config import INPUT_COLS, PAST_COLS


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
        "NeuralForecast_Nhits": NeuralForecastModel(
            model_name=f"NFTNhits_I{input_chunk_length}_O{output_chunk_length}",
            model="NHITS",
            save_checkpoints=True,
            force_reset=True,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=5,
            batch_size=128,
            optimizer_kwargs={"lr": 1e-3},
            pl_trainer_kwargs={
                "logger": CSVLogger(f"outputs/logs/I{input_chunk_length}_O{output_chunk_length}", name="NF_NHITS"),
                "log_every_n_steps": 1,
                "enable_model_summary": False,
            }
        ),
        "NeuralForecast_NLinear": NeuralForecastModel(
            model_name=f"NFTNLinear_I{input_chunk_length}_O{output_chunk_length}",
            model="NLinear",
            save_checkpoints=True,
            force_reset=True,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=5,
            batch_size=128,
            optimizer_kwargs={"lr": 1e-3},
            pl_trainer_kwargs={
                "logger": CSVLogger(f"outputs/logs/I{input_chunk_length}_O{output_chunk_length}", name="NF_NLinear"),
                "log_every_n_steps": 1,
                "enable_model_summary": False,
            }
        ),
        "NeuralForecast_DLinear": NeuralForecastModel(
            model_name=f"NFTDLinear_I{input_chunk_length}_O{output_chunk_length}",
            model="DLinear",
            save_checkpoints=True,
            force_reset=True,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=5,
            batch_size=128,
            optimizer_kwargs={"lr": 1e-3},
            pl_trainer_kwargs={
                "logger": CSVLogger(f"outputs/logs/I{input_chunk_length}_O{output_chunk_length}", name="NF_DLinear"),
                "log_every_n_steps": 1,
                "enable_model_summary": False,
            }
        ),
        "NeuralForecast_TSMixer": NeuralForecastModel(
            model_name=f"NFTTSMixer_I{input_chunk_length}_O{output_chunk_length}",
            model="TSMixer",
            save_checkpoints=True,
            force_reset=True,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=10,
            batch_size=128,
            optimizer_kwargs={"lr": 1e-3},
            pl_trainer_kwargs={
                "logger": CSVLogger(f"outputs/logs/I{input_chunk_length}_O{output_chunk_length}", name="NF_TSMixer"),
                "log_every_n_steps": 1,
                "enable_model_summary": False,
            }
        ),
        "NeuralForecast_TFT": NeuralForecastModel(
            model_name=f"NFTTFT_I{input_chunk_length}_O{output_chunk_length}",
            model="TFT",
            save_checkpoints=True,
            force_reset=True,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=10,
            batch_size=128,
            optimizer_kwargs={"lr": 1e-3},
            pl_trainer_kwargs={
                "logger": CSVLogger(f"outputs/logs/I{input_chunk_length}_O{output_chunk_length}", name="NF_TFT"),
                "log_every_n_steps": 1,
                "enable_model_summary": False,
            }
        ),
    }

    return models
