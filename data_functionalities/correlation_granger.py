from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.stattools import grangercausalitytests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_COLS

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MAX_LAG = 30
DEFAULT_ALPHA = 0.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-session cross-correlation and Granger causality for time series data. "
            "Each `session_index` is treated as a separate time series."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=PROJECT_ROOT / "data_preprocessed" / "data1.csv",
        help="Path to a CSV file containing `session_index` and `sequence_index`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where result CSV files will be saved.",
    )
    parser.add_argument(
        "--x-col",
        action="append",
        default=None,
        help=(
            "Column name for X variable. Can be provided multiple times. "
            "If omitted together with --y-col, pairs are built from RAW_COLS x RAW_COLS."
        ),
    )
    parser.add_argument(
        "--y-col",
        action="append",
        default=None,
        help=(
            "Column name for Y variable. Can be provided multiple times. "
            "Must be used together with --x-col (unless both omitted)."
        ),
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=DEFAULT_MAX_LAG,
        help="Maximum lag for cross-correlation and Granger tests.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Significance threshold for Granger causality.",
    )
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _build_pairs(df: pd.DataFrame, x_cols: list[str] | None, y_cols: list[str] | None) -> list[tuple[str, str]]:
    if (x_cols is None) != (y_cols is None):
        raise ValueError("Use --x-col and --y-col together, or omit both.")

    if x_cols is not None and y_cols is not None:
        pairs = list(itertools.product(x_cols, y_cols))
        return [(x_col, y_col) for x_col, y_col in pairs if x_col != y_col]

    return [
        (x_col, y_col)
        for x_col, y_col in itertools.product(RAW_COLS, RAW_COLS)
        if x_col in df.columns and y_col in df.columns and x_col != y_col
    ]


def _aligned_corr(x_values: np.ndarray, y_values: np.ndarray, lag: int) -> tuple[float, int]:
    if lag < 0:
        shifted = -lag
        x_aligned = x_values[shifted:]
        y_aligned = y_values[:-shifted]
    elif lag > 0:
        x_aligned = x_values[:-lag]
        y_aligned = y_values[lag:]
    else:
        x_aligned = x_values
        y_aligned = y_values

    n_obs = int(len(x_aligned))
    if n_obs < 3:
        return np.nan, n_obs

    x_std = float(np.std(x_aligned, ddof=0))
    y_std = float(np.std(y_aligned, ddof=0))
    if np.isclose(x_std, 0.0) or np.isclose(y_std, 0.0):
        return np.nan, n_obs

    corr = float(np.corrcoef(x_aligned, y_aligned)[0, 1])
    return corr, n_obs


def _compute_cross_correlation(
    x_values: np.ndarray,
    y_values: np.ndarray,
    max_lag: int,
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    lag_rows: list[dict[str, float | int]] = []

    for lag in range(-max_lag, max_lag + 1):
        corr, n_obs = _aligned_corr(x_values, y_values, lag)
        lag_rows.append({"lag": lag, "corr": corr, "n_obs": n_obs})

    valid_lags = [row for row in lag_rows if not pd.isna(row["corr"])]
    if not valid_lags:
        summary = {
            "best_corr_lag": np.nan,
            "best_corr": np.nan,
            "best_corr_abs": np.nan,
        }
        return lag_rows, summary

    best_row = max(valid_lags, key=lambda row: abs(float(row["corr"])))
    summary = {
        "best_corr_lag": int(best_row["lag"]),
        "best_corr": float(best_row["corr"]),
        "best_corr_abs": abs(float(best_row["corr"])),
    }
    return lag_rows, summary


def _run_granger_tests(target: np.ndarray, cause: np.ndarray, max_lag: int, alpha: float) -> dict[str, float | int | bool | str]:
    paired = pd.DataFrame({"target": target, "cause": cause}).dropna().values

    if len(paired) <= max_lag + 1:
        return {
            "granger_min_p": np.nan,
            "granger_best_lag": np.nan,
            "granger_significant": False,
            "granger_error": f"Too few observations ({len(paired)}) for max_lag={max_lag}",
        }

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="verbose is deprecated since functions should not print results",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="covariance of constraints does not have full rank.*",
                category=ValueWarning,
            )
            result = grangercausalitytests(paired, maxlag=max_lag, verbose=False)
    except Exception as exc:
        return {
            "granger_min_p": np.nan,
            "granger_best_lag": np.nan,
            "granger_significant": False,
            "granger_error": str(exc),
        }

    lag_p_values: list[tuple[int, float]] = []
    for lag, lag_result in result.items():
        p_value = float(lag_result[0]["ssr_ftest"][1])
        lag_p_values.append((int(lag), p_value))

    best_lag, min_p = min(lag_p_values, key=lambda item: item[1])
    return {
        "granger_min_p": min_p,
        "granger_best_lag": best_lag,
        "granger_significant": bool(min_p < alpha),
        "granger_error": "",
    }


def _aggregate_pair_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "x_col",
                "y_col",
                "sessions_total",
                "sessions_with_valid_corr",
                "valid_corr_ratio",
                "mean_best_corr",
                "mean_best_corr_abs",
                "median_best_corr_abs",
                "max_best_corr_abs",
                "weighted_mean_best_corr_abs",
                "most_common_best_lag",
                "x_to_y_significant_ratio",
                "y_to_x_significant_ratio",
                "bidirectional_significant_ratio",
            ]
        )

    def _most_common_lag(series: pd.Series) -> float:
        clean = series.dropna()
        if clean.empty:
            return np.nan
        mode = clean.mode()
        if mode.empty:
            return np.nan
        return float(mode.iloc[0])

    grouped = summary_df.groupby(["x_col", "y_col"], sort=True)
    aggregated = grouped.agg(
        sessions_total=("session_index", "count"),
        sessions_with_valid_corr=("best_corr", lambda s: int(s.notna().sum())),
        mean_best_corr=("best_corr", "mean"),
        mean_best_corr_abs=("best_corr_abs", "mean"),
        median_best_corr_abs=("best_corr_abs", "median"),
        max_best_corr_abs=("best_corr_abs", "max"),
        x_to_y_significant_ratio=("x_to_y_significant", "mean"),
        y_to_x_significant_ratio=("y_to_x_significant", "mean"),
        bidirectional_significant_ratio=(
            "x_to_y_significant",
            lambda s: float((s & summary_df.loc[s.index, "y_to_x_significant"]).mean()),
        ),
        most_common_best_lag=("best_corr_lag", _most_common_lag),
    ).reset_index()

    weighted_source = summary_df.dropna(subset=["best_corr_abs", "n_obs"]).copy()
    weighted_source["weighted_corr"] = weighted_source["best_corr_abs"] * weighted_source["n_obs"]
    weighted = (
        weighted_source.groupby(["x_col", "y_col"], sort=True)[["weighted_corr", "n_obs"]]
        .sum()
        .reset_index()
    )
    weighted["weighted_mean_best_corr_abs"] = weighted["weighted_corr"] / weighted["n_obs"]
    weighted = weighted[["x_col", "y_col", "weighted_mean_best_corr_abs"]]

    aggregated = aggregated.merge(weighted, on=["x_col", "y_col"], how="left")
    aggregated["valid_corr_ratio"] = (
        aggregated["sessions_with_valid_corr"] / aggregated["sessions_total"]
    )

    return aggregated.sort_values(
        by=["mean_best_corr_abs", "weighted_mean_best_corr_abs", "max_best_corr_abs"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def analyze_file(
    input_file: Path,
    output_dir: Path,
    pairs: list[tuple[str, str]],
    max_lag: int,
    alpha: float,
) -> tuple[Path, Path, Path]:
    data = pd.read_csv(input_file)
    _validate_columns(data, ["session_index", "sequence_index"])

    for column in {column for pair in pairs for column in pair}:
        if column not in data.columns:
            raise ValueError(f"Column `{column}` is not present in {input_file.name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    lag_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    grouped = data.sort_values(["session_index", "sequence_index"]).groupby("session_index", sort=True)
    for session_index, session_df in grouped:
        for x_col, y_col in pairs:
            subset = session_df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(subset) < 3:
                summary_rows.append(
                    {
                        "session_index": session_index,
                        "x_col": x_col,
                        "y_col": y_col,
                        "n_obs": len(subset),
                        "best_corr_lag": np.nan,
                        "best_corr": np.nan,
                        "best_corr_abs": np.nan,
                        "x_to_y_min_p": np.nan,
                        "x_to_y_best_lag": np.nan,
                        "x_to_y_significant": False,
                        "x_to_y_error": "Too few observations",
                        "y_to_x_min_p": np.nan,
                        "y_to_x_best_lag": np.nan,
                        "y_to_x_significant": False,
                        "y_to_x_error": "Too few observations",
                    }
                )
                continue

            x_values = subset[x_col].values
            y_values = subset[y_col].values

            pair_lag_rows, corr_summary = _compute_cross_correlation(x_values, y_values, max_lag=max_lag)
            for row in pair_lag_rows:
                lag_rows.append(
                    {
                        "session_index": session_index,
                        "x_col": x_col,
                        "y_col": y_col,
                        "lag": row["lag"],
                        "corr": row["corr"],
                        "n_obs": row["n_obs"],
                    }
                )

            x_to_y = _run_granger_tests(target=y_values, cause=x_values, max_lag=max_lag, alpha=alpha)
            y_to_x = _run_granger_tests(target=x_values, cause=y_values, max_lag=max_lag, alpha=alpha)

            summary_rows.append(
                {
                    "session_index": session_index,
                    "x_col": x_col,
                    "y_col": y_col,
                    "n_obs": len(subset),
                    "best_corr_lag": corr_summary["best_corr_lag"],
                    "best_corr": corr_summary["best_corr"],
                    "best_corr_abs": corr_summary["best_corr_abs"],
                    "x_to_y_min_p": x_to_y["granger_min_p"],
                    "x_to_y_best_lag": x_to_y["granger_best_lag"],
                    "x_to_y_significant": x_to_y["granger_significant"],
                    "x_to_y_error": x_to_y["granger_error"],
                    "y_to_x_min_p": y_to_x["granger_min_p"],
                    "y_to_x_best_lag": y_to_x["granger_best_lag"],
                    "y_to_x_significant": y_to_x["granger_significant"],
                    "y_to_x_error": y_to_x["granger_error"],
                }
            )

    stem = input_file.stem
    summary_path = output_dir / f"cross_corr_granger_summary_{stem}.csv"
    lags_path = output_dir / f"cross_corr_lags_{stem}.csv"
    aggregate_path = output_dir / f"cross_corr_granger_aggregate_{stem}.csv"

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(lag_rows).to_csv(lags_path, index=False)
    _aggregate_pair_summary(summary_df).to_csv(aggregate_path, index=False)

    return summary_path, lags_path, aggregate_path


def main() -> None:
    args = _parse_args()

    if args.max_lag < 1:
        raise ValueError("--max-lag must be >= 1")
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("--alpha must be in (0, 1)")

    input_file: Path = args.input_file
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    data = pd.read_csv(input_file, nrows=5000)
    pairs = _build_pairs(data, args.x_col, args.y_col)
    if not pairs:
        raise ValueError("No variable pairs selected. Provide valid --x-col and --y-col values.")

    summary_path, lags_path, aggregate_path = analyze_file(
        input_file=input_file,
        output_dir=args.output_dir,
        pairs=pairs,
        max_lag=args.max_lag,
        alpha=args.alpha,
    )

    print(f"Analysis completed for {input_file}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved lag-level cross-correlation: {lags_path}")
    print(f"Saved aggregated pair ranking: {aggregate_path}")
    print(f"Pairs analyzed: {len(pairs)}")


if __name__ == "__main__":
    main()
