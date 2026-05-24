#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute min and max values for numeric columns across all CSV files "
            "in a directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data_preprocessed"),
        help="Directory containing preprocessed CSV files (default: data_preprocessed).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for input files (default: *.csv).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs/min_max_data_preprocessed.csv"),
        help="Output CSV file path (default: outputs/min_max_data_preprocessed.csv).",
    )
    return parser.parse_args()


def load_numeric_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.loc[:, numeric_df.notna().any()]
    return numeric_df


def update_global_min_max(
    global_min: pd.Series | None,
    global_max: pd.Series | None,
    numeric_df: pd.DataFrame,
) -> tuple[pd.Series | None, pd.Series | None]:
    if numeric_df.empty:
        return global_min, global_max

    file_min = numeric_df.min()
    file_max = numeric_df.max()

    if global_min is None:
        global_min = file_min.copy()
    else:
        global_min = pd.concat([global_min, file_min], axis=1).min(axis=1)

    if global_max is None:
        global_max = file_max.copy()
    else:
        global_max = pd.concat([global_max, file_max], axis=1).max(axis=1)

    return global_min, global_max


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_file = args.output_file

    csv_files = sorted(input_dir.glob(args.pattern))
    if not csv_files:
        raise SystemExit(f"No files matched {args.pattern!r} in {input_dir}")

    global_min: pd.Series | None = None
    global_max: pd.Series | None = None
    for csv_path in csv_files:
        numeric_df = load_numeric_frame(csv_path)
        global_min, global_max = update_global_min_max(global_min, global_max, numeric_df)

    if global_min is None or global_max is None:
        result = pd.DataFrame(columns=["column", "min", "max"])
    else:
        result = pd.DataFrame({
            "column": global_min.index,
            "min": global_min.values,
            "max": global_max.values,
        })

    result = result.sort_values("column").reset_index(drop=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)

    print(f"Processed {len(csv_files)} files.")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
