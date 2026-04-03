from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def find_first_csv(data_dir: Path) -> Path:
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return csv_files[0]


def analyze_gaps(file_path: Path, timestamp_column: str, expected_step_seconds: int) -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(file_path, usecols=[timestamp_column])
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce")
    frame = frame.dropna(subset=[timestamp_column]).copy()

    if frame.empty:
        raise ValueError(f"No valid timestamps found in column '{timestamp_column}'")

    frame = frame.sort_values(timestamp_column).reset_index(drop=True)
    frame["prev_time"] = frame[timestamp_column].shift(1)
    frame["delta_seconds"] = (frame[timestamp_column] - frame["prev_time"]).dt.total_seconds()

    gaps = frame[(frame["prev_time"].notna()) & (frame["delta_seconds"] != expected_step_seconds)].copy()

    gaps["missing_points"] = gaps["delta_seconds"].apply(
        lambda seconds: max(int(round(seconds / expected_step_seconds)) - 1, 0)
        if seconds > expected_step_seconds
        else 0
    )

    return gaps[["prev_time", timestamp_column, "delta_seconds", "missing_points"]], len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze timestamp gaps in the first CSV file from a data directory."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory with CSV files")
    parser.add_argument("--timestamp-column", default="time", help="Timestamp column name")
    parser.add_argument(
        "--expected-step-seconds",
        type=int,
        default=10,
        help="Expected interval between datapoints in seconds",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/first_file_time_gaps.csv"),
        help="Where to save detailed gap rows",
    )
    args = parser.parse_args()

    first_file = find_first_csv(args.data_dir)
    gaps, total_rows = analyze_gaps(first_file, args.timestamp_column, args.expected_step_seconds)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(args.output_csv, index=False)

    print(f"Analyzed file: {first_file}")
    print(f"Total valid rows: {total_rows}")
    print(f"Expected step: {args.expected_step_seconds} seconds")
    print(f"Rows with non-{args.expected_step_seconds}s interval: {len(gaps)}")
    print(f"Saved gap details: {args.output_csv}")

    if gaps.empty:
        print("No gaps detected.")
        return

    print("\nTop 10 largest intervals:")
    print(gaps.sort_values("delta_seconds", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()