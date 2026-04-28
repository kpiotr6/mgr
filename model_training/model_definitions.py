"""
Model definitions for time series forecasting.
Centralizes model configuration and instantiation logic.
"""

from darts.models import (
    NaiveSeasonal,
    LinearRegressionModel,
    NLinearModel,
    DLinearModel,
    TSMixerModel,
    NHiTSModel,
    LightGBMModel,
    CatBoostModel,
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
        # "LightGBM": LightGBMModel(
        #     lags=input_chunk_length,
        #     lags_future_covariates=(input_chunk_length, output_chunk_length),
        #     output_chunk_length=output_chunk_length
        # ),
        # "CatBoost": CatBoostModel(
        #     lags=input_chunk_length,
        #     lags_future_covariates=(input_chunk_length, output_chunk_length),
        #     output_chunk_length=output_chunk_length
        # ),
        # "NLinear": NLinearModel(
        #     model_name=f"NLinear_I{input_chunk_length}_O{output_chunk_length}",
        #     save_checkpoints=True,
        #     force_reset=True,
        #     input_chunk_length=input_chunk_length,
        #     output_chunk_length=output_chunk_length,
        #     const_init=False,
        #     n_epochs=40,
        #     batch_size=128,
        #     optimizer_kwargs={"lr": 1e-3},
        #     pl_trainer_kwargs={
        #         "logger": CSVLogger(f"outputs/logs/I{input_chunk_length}_O{output_chunk_length}", name="NLinear"),
        #         "log_every_n_steps": 1
        #     }
        # ),
        # "DLinear": DLinearModel(
        #     model_name=f"DLinear_I{input_chunk_length}_O{output_chunk_length}",
        #     save_checkpoints=True,
        #     force_reset=True,
        #     input_chunk_length=input_chunk_length,
        #     output_chunk_length=output_chunk_length,
        #     const_init=False,
        #     n_epochs=40,
        #     batch_size=128,
        #     optimizer_kwargs={"lr": 1e-3},
        #     pl_trainer_kwargs={
        #         "logger": CSVLogger(f"outputs/logs/I{input_chunk_length}_O{output_chunk_length}", name="NLinear"),
        #         "log_every_n_steps": 1,
        #         "enable_model_summary": False,
        #         "enable_progress_bar": False
        #     }
        # ),
        # "TSMixer": TSMixerModel(
        #     model_name=f"TSMixer_I{input_chunk_length}_O{output_chunk_length}",
        #     save_checkpoints=True,
        #     force_reset=True,
        #     input_chunk_length=input_chunk_length,
        #     output_chunk_length=output_chunk_length,
        #     hidden_size=128,
        #     ff_size=64,
        #     num_blocks=3,
        #     dropout=0.1,
        #     n_epochs=20,
        #     batch_size=128,
        #     optimizer_kwargs={"lr": 1e-3},
        #     # Removed custom torch_metrics
        #     pl_trainer_kwargs={
        #         "logger": CSVLogger(f"outputs/logs/I{input_chunk_length}_O{output_chunk_length}", name="TSMixer"),
        #         "log_every_n_steps": 1,
        #         "enable_model_summary": False,
        #         # "enable_progress_bar": False
        #     }
        # ),
        # "NHiTS": NHiTSModel(
        #     model_name=f"NHiTS_I{input_chunk_length}_O{output_chunk_length}",
        #     save_checkpoints=True,
        #     force_reset=True,
        #     input_chunk_length=input_chunk_length,
        #     output_chunk_length=output_chunk_length,
        #     n_epochs=20,
        #     batch_size=128,
        #     optimizer_kwargs={"lr": 1e-3},
        #     pl_trainer_kwargs={
        #         "logger": CSVLogger(f"outputs/logs/I{input_chunk_length}_O{output_chunk_length}", name="NHiTS"),
        #         "log_every_n_steps": 1,
        #         "enable_model_summary": False,
        #         # "enable_progress_bar": False
        #     }
        # ),
    }

    return models
