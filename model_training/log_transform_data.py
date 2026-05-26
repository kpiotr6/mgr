"""
Module for log-transforming time series data.
Provides utilities to transform TimeSeries objects with log1p,
and reverse the transformation using expm1.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from darts import TimeSeries


class LogTransformStorage:
    """Storage for log-transform parameters used for reverse transformation."""

    def __init__(self):
        self.params: Dict[int, Dict[str, Dict[str, float]]] = {}

    def store_params(self, ts_index: int, params_dict: Dict[str, Dict[str, float]]):
        """Store parameters for a specific time series index, merging with existing columns."""
        if ts_index not in self.params:
            self.params[ts_index] = {}
        self.params[ts_index].update(params_dict)

    def get_column_params(self, ts_index: int, column: str) -> Dict[str, float]:
        """Retrieve stored parameters for a specific series index and column."""
        return self.params.get(ts_index, {}).get(column)


_log_transform_storage = LogTransformStorage()


def get_log_transform_storage() -> LogTransformStorage:
    """Get global log-transform storage object."""
    return _log_transform_storage


def reset_log_transform_storage():
    """Reset global log-transform storage object."""
    global _log_transform_storage
    _log_transform_storage = LogTransformStorage()


def log_transform_timeseries(
    ts_list: List[TimeSeries],
    store_params: bool = True,
    min_margin: float = 1e-6,
) -> List[TimeSeries]:
    """
    Apply a safe log1p transform to TimeSeries values.

    For each column, if values contain entries <= -1, the whole column is shifted
    upward by `-min_value + min_margin` so `log1p` stays valid.
    """
    transformed = []

    for ts_idx, ts in enumerate(ts_list):
        df = ts.to_dataframe()
        transformed_df = df.copy()
        params_dict: Dict[str, Dict[str, float]] = {}

        for col in df.columns:
            values = df[col].values.astype(float)
            if len(values) == 0:
                if store_params:
                    params_dict[col] = {"shift": 0.0}
                continue

            min_value = float(np.nanmin(values))
            shift = 0.0

            if min_value <= -1.0:
                shift = -min_value + min_margin

            transformed_df[col] = np.log1p(values + shift)

            if store_params:
                params_dict[col] = {"shift": shift}

        if store_params:
            _log_transform_storage.store_params(ts_idx, params_dict)

        transformed.append(TimeSeries.from_dataframe(transformed_df.astype("float64")))

    return transformed


def reverse_log_transform_timeseries(ts_list: List[TimeSeries], ts_indices: List[int] = None) -> List[TimeSeries]:
    """
    Reverse log-transform by applying expm1 and subtracting stored shifts.

    Args:
        ts_list: List of log-transformed TimeSeries objects
        ts_indices: Original series indices used during transform

    Returns:
        List of reverse-transformed TimeSeries objects
    """
    if ts_indices is None:
        ts_indices = list(range(len(ts_list)))

    reversed_list = []

    for ts_obj, ts_idx in zip(ts_list, ts_indices):
        df = ts_obj.to_dataframe()
        reversed_df = df.copy()

        for col in df.columns:
            params = _log_transform_storage.get_column_params(ts_idx, col)
            if params is None:
                continue

            shift = float(params.get("shift", 0.0))
            reversed_df[col] = np.expm1(df[col].values) - shift

        reversed_list.append(TimeSeries.from_dataframe(reversed_df.astype("float64")))

    return reversed_list


def reverse_log_transform_dataframe(df: pd.DataFrame, ts_index: int, columns: List[str] = None) -> pd.DataFrame:
    """
    Reverse log-transform on a pandas DataFrame.

    Args:
        df: DataFrame with log-transformed values
        ts_index: Original series index used during transform
        columns: Columns to reverse (defaults to all numeric columns)

    Returns:
        DataFrame in original value space
    """
    reversed_df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        params = _log_transform_storage.get_column_params(ts_index, col)
        if params is None or col not in reversed_df.columns:
            continue

        shift = float(params.get("shift", 0.0))
        reversed_df[col] = np.expm1(reversed_df[col].values) - shift

    return reversed_df