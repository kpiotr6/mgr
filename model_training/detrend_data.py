"""
Module for detrending time series data.
Provides utilities to detrend TimeSeries objects using various methods.
Supports both forward detrending and reverse (retrending) operations.
"""

from typing import List, Tuple, Dict
from darts import TimeSeries
from scipy import signal
import numpy as np
import pandas as pd


class DetrenderStorage:
    """
    Storage class to keep track of trend information for reverse detrending.
    """
    def __init__(self):
        self.trends: Dict[int, Dict[str, np.ndarray]] = {}

    def store_trend(self, ts_index: int, trends_dict: Dict[str, np.ndarray]):
        """Store trend information for a specific time series."""
        if ts_index not in self.trends:
            self.trends[ts_index] = {}
        self.trends[ts_index].update(trends_dict)

    def get_trend(self, ts_index: int, column: str) -> np.ndarray:
        """Retrieve trend for a specific time series and column."""
        return self.trends.get(ts_index, {}).get(column)


# Global storage for detrending information
_detrend_storage = DetrenderStorage()


def get_detrend_storage() -> DetrenderStorage:
    """Get the global detrend storage object."""
    return _detrend_storage


def reset_detrend_storage():
    """Reset the global detrend storage."""
    global _detrend_storage
    _detrend_storage = DetrenderStorage()


def detrend_timeseries_linear(ts_list: List[TimeSeries], store_trends: bool = True) -> List[TimeSeries]:
    """
    Detrend a list of TimeSeries objects using linear detrending.
    Optionally stores trend information for later reverse detrending.

    Args:
        ts_list: List of TimeSeries objects to detrend
        store_trends: Whether to store trend info for reverse detrending (default: True)

    Returns:
        List of detrended TimeSeries objects
    """
    detrended = []
    for ts_idx, ts in enumerate(ts_list):
        df = ts.to_dataframe()

        # Apply linear detrending to each column
        detrended_df = df.copy()
        trends_dict = {}

        for col in df.columns:
            values = df[col].values
            if len(values) > 1:
                # Use scipy's linear detrending and get the trend
                detrended_values = signal.detrend(values, type='linear')
                trend = values - detrended_values

                detrended_df[col] = detrended_values
                if store_trends:
                    trends_dict[col] = trend
            else:
                if store_trends:
                    trends_dict[col] = np.zeros_like(values)

        if store_trends:
            _detrend_storage.store_trend(ts_idx, trends_dict)

        # Convert back to TimeSeries
        detrended_ts = TimeSeries.from_dataframe(detrended_df.astype("float64"))
        detrended.append(detrended_ts)

    return detrended


def detrend_timeseries_polynomial(ts_list: List[TimeSeries], order: int = 2, store_trends: bool = True) -> List[TimeSeries]:
    """
    Detrend a list of TimeSeries objects using polynomial detrending.
    Optionally stores trend information for later reverse detrending.

    Args:
        ts_list: List of TimeSeries objects to detrend
        order: Order of the polynomial for detrending (default: 2)
        store_trends: Whether to store trend info for reverse detrending (default: True)

    Returns:
        List of detrended TimeSeries objects
    """
    detrended = []
    for ts_idx, ts in enumerate(ts_list):
        df = ts.to_dataframe()

        # Apply polynomial detrending to each column
        detrended_df = df.copy()
        trends_dict = {}

        for col in df.columns:
            values = df[col].values
            if len(values) > order:
                # Use scipy's polynomial detrending and get the trend
                detrended_values = signal.detrend(values, type='polynomial', order=order)
                trend = values - detrended_values

                detrended_df[col] = detrended_values
                if store_trends:
                    trends_dict[col] = trend
            else:
                if store_trends:
                    trends_dict[col] = np.zeros_like(values)

        if store_trends:
            _detrend_storage.store_trend(ts_idx, trends_dict)

        # Convert back to TimeSeries
        detrended_ts = TimeSeries.from_dataframe(detrended_df.astype("float64"))
        detrended.append(detrended_ts)

    return detrended


def detrend_timeseries_constant(ts_list: List[TimeSeries], store_trends: bool = True) -> List[TimeSeries]:
    """
    Detrend a list of TimeSeries objects using constant (mean) detrending.
    Optionally stores trend information for later reverse detrending.

    Args:
        ts_list: List of TimeSeries objects to detrend
        store_trends: Whether to store trend info for reverse detrending (default: True)

    Returns:
        List of detrended TimeSeries objects
    """
    detrended = []
    for ts_idx, ts in enumerate(ts_list):
        df = ts.to_dataframe()

        # Apply constant detrending (remove mean) to each column
        detrended_df = df.copy()
        trends_dict = {}

        for col in df.columns:
            values = df[col].values
            if len(values) > 0:
                # Use scipy's constant detrending and get the trend
                detrended_values = signal.detrend(values, type='constant')
                trend = values - detrended_values

                detrended_df[col] = detrended_values
                if store_trends:
                    trends_dict[col] = trend
            else:
                if store_trends:
                    trends_dict[col] = np.zeros_like(values)

        if store_trends:
            _detrend_storage.store_trend(ts_idx, trends_dict)

        # Convert back to TimeSeries
        detrended_ts = TimeSeries.from_dataframe(detrended_df.astype("float64"))
        detrended.append(detrended_ts)

    return detrended


def reverse_detrend_timeseries(ts_list: List[TimeSeries], ts_indices: List[int] = None) -> List[TimeSeries]:
    """
    Reverse the detrending process by adding back the stored trends.

    Args:
        ts_list: List of detrended TimeSeries objects to reverse
        ts_indices: List of indices corresponding to the original time series (if None, uses sequential indices)

    Returns:
        List of retrended TimeSeries objects
    """
    if ts_indices is None:
        ts_indices = list(range(len(ts_list)))

    retrended = []
    for pred_ts, ts_idx in zip(ts_list, ts_indices):
        df = pred_ts.to_dataframe()
        retrended_df = df.copy()

        trends_dict = _detrend_storage.trends.get(ts_idx, {})

        for col in df.columns:
            if col in trends_dict:
                trend = trends_dict[col]
                values = df[col].values

                # Handle length mismatch: if prediction is shorter, use the trend part
                if len(values) < len(trend):
                    # Predictions are typically at the end, so use the tail of the trend
                    retrended_values = values + trend[-len(values):]
                else:
                    # If prediction is longer (shouldn't happen), repeat last trend value
                    retrended_values = values + trend

                retrended_df[col] = retrended_values

        retrended_ts = TimeSeries.from_dataframe(retrended_df.astype("float64"))
        retrended.append(retrended_ts)

    return retrended


def reverse_detrend_dataframe(df: pd.DataFrame, ts_index: int, columns: List[str] = None) -> pd.DataFrame:
    """
    Reverse detrending on a dataframe (useful for prediction outputs).

    Args:
        df: DataFrame with detrended values
        ts_index: Index of the original time series
        columns: List of columns to retrend (if None, retrends all numeric columns)

    Returns:
        DataFrame with retrended values
    """
    retrended_df = df.copy()
    trends_dict = _detrend_storage.trends.get(ts_index, {})

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col in trends_dict and col in df.columns:
            trend = trends_dict[col]
            values = df[col].values

            # Handle length mismatch
            if len(values) < len(trend):
                retrended_values = values + trend[-len(values):]
            else:
                retrended_values = values + trend

            retrended_df[col] = retrended_values

    return retrended_df
