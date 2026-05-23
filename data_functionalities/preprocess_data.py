from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import INPUT_COLS, PAST_COLS, SESSION_COL, STATIC_COLS, TARGET_COLS, TIME_COL


def _build_output_columns() -> list[str]:
    output_columns: list[str] = [SESSION_COL, TIME_COL]
    for column in INPUT_COLS + PAST_COLS + STATIC_COLS + TARGET_COLS:
        if column not in output_columns:
            output_columns.append(column)
    return output_columns


OUTPUT_COLS = _build_output_columns()


def _normalize_session_frame(session_df: pd.DataFrame) -> pd.DataFrame:
    session_df = session_df.copy()
    session_df = session_df.sort_values(TIME_COL).reset_index(drop=True)

    original_sequence = pd.to_numeric(session_df[TIME_COL], errors="coerce")
    if original_sequence.isna().any():
        raise ValueError("`sequence_index` contains non-numeric values.")

    original_sequence = original_sequence.astype(int)
    sequence_offset = int(original_sequence.iloc[0]) - 1
    normalized_sequence = original_sequence - sequence_offset
    session_df[TIME_COL] = normalized_sequence.astype(int)

    full_index = pd.RangeIndex(start=1, stop=int(normalized_sequence.max()) + 1, step=1)
    session_df = session_df.set_index(TIME_COL).reindex(full_index)

    session_value = session_df[SESSION_COL].dropna().iloc[0]
    session_df[SESSION_COL] = session_value

    session_df[TIME_COL] = session_df.index.astype(int)
    session_df = session_df.ffill()

    session_df[TIME_COL] = session_df.index.astype(int)
    session_df[SESSION_COL] = session_value
    session_df = session_df.reset_index(drop=True)
    return session_df


def preprocess_file(input_file: Path, output_file: Path) -> tuple[int, int]:
    df = pd.read_csv(input_file)

    missing_columns = [column for column in OUTPUT_COLS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{input_file.name} is missing columns: {missing_columns}")

    df = df[OUTPUT_COLS].copy()
    df[SESSION_COL] = pd.to_numeric(df[SESSION_COL], errors="coerce")
    if df[SESSION_COL].isna().any():
        raise ValueError(f"{input_file.name} contains non-numeric `session_index` values.")
    df[SESSION_COL] = df[SESSION_COL].astype(int)

    processed_sessions = []
    for _, session_df in df.groupby(SESSION_COL, sort=True):
        session_df = session_df.sort_values(TIME_COL).reset_index(drop=True)
        zero_mask = session_df["gran1_blain"].eq(0)
        if zero_mask.any():
            first_zero_position = int(zero_mask.idxmax())
            session_df = session_df.iloc[:first_zero_position].copy()

        if session_df.empty:
            continue

        processed_sessions.append(_normalize_session_frame(session_df))

    if not processed_sessions:
        raise ValueError(
            f"{input_file.name} has no session rows left after applying `gran1_blain == 0` truncation."
        )

    processed_df = pd.concat(processed_sessions, ignore_index=True)
    processed_df = processed_df[OUTPUT_COLS]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_file, index=False)

    return len(df), len(processed_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw data files into `data_preprocessed`.")
    parser.add_argument("--input-dir", default="data", help="Folder with raw CSV files.")
    parser.add_argument("--output-dir", default="data_preprocessed", help="Folder for processed CSV files.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    for input_file in csv_files:
        output_file = output_dir / input_file.name
        original_rows, processed_rows = preprocess_file(input_file, output_file)
        print(f"{input_file.name}: {original_rows} -> {processed_rows} rows saved to {output_file}")


if __name__ == "__main__":
    main()