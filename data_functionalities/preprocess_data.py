from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import INPUT_COLS, PAST_COLS, SESSION_COL, TARGET_COLS, TIME_COL


def _build_output_columns() -> list[str]:
    output_columns: list[str] = [SESSION_COL, TIME_COL]
    for column in INPUT_COLS + PAST_COLS + TARGET_COLS:
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


def preprocess_file(
    input_file: Path,
    output_file: Path,
    ma_window: int = 0,
    subsample_window: int = 0,
    check_gran_empty: bool = False,
    chunk_size: int = 0
) -> tuple[int, int, pd.Series]:
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
    new_session_id = 1  # Used to ensure unique IDs if we split sequences

    for _, session_df in df.groupby(SESSION_COL, sort=True):
        session_df = session_df.sort_values(TIME_COL).reset_index(drop=True)

        # Truncate session if gran1_blain is empty (NaN) or 0, based on the flag
        if check_gran_empty:
            empty_mask = session_df["gran1_blain"].eq(0) | session_df["gran1_blain"].isna()
            if empty_mask.any():
                first_empty_position = int(empty_mask.idxmax())
                session_df = session_df.iloc[:first_empty_position].copy()

        if session_df.empty:
            continue

        session_df = _normalize_session_frame(session_df)

        if subsample_window > 1:
            subsampled_sessions = session_df.groupby(session_df.index // subsample_window, sort=True).mean(numeric_only=True)
            subsampled_sessions[SESSION_COL] = int(session_df[SESSION_COL].iloc[0])
            subsampled_sessions[TIME_COL] = range(1, len(subsampled_sessions) + 1)
            session_df = subsampled_sessions[OUTPUT_COLS].reset_index(drop=True)

        # Split into smaller chunks if requested
        if chunk_size > 0:
            num_chunks = len(session_df) // chunk_size
            for i in range(num_chunks):
                chunk = session_df.iloc[i * chunk_size : (i + 1) * chunk_size].copy()
                chunk[SESSION_COL] = new_session_id
                chunk[TIME_COL] = range(1, chunk_size + 1)
                new_session_id += 1
                processed_sessions.append(chunk)
        else:
            processed_sessions.append(session_df)

    if not processed_sessions:
        raise ValueError(
            f"{input_file.name} has no session rows left after processing."
        )

    processed_df = pd.concat(processed_sessions, ignore_index=True)
    processed_df = processed_df[OUTPUT_COLS]

    # Calculate moving average if a window length is provided
    if ma_window > 0:
        cols_to_ma = [col for col in OUTPUT_COLS if col not in (SESSION_COL, TIME_COL)]
        for col in cols_to_ma:
            ma_col_name = f"{col}_ma_{ma_window}"
            processed_df[ma_col_name] = (
                processed_df.groupby(SESSION_COL)[col]
                .transform(lambda x: x.rolling(window=ma_window, min_periods=1).mean())
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_file, index=False)

    # Return series containing length of each session sequence
    session_lengths = processed_df.groupby(SESSION_COL).size()

    return len(df), len(processed_df), session_lengths


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw data files into `data_preprocessed`.")
    parser.add_argument("--input-dir", default="data", help="Folder with raw CSV files.")
    parser.add_argument("--output-dir", default="data_preprocessed", help="Folder for processed CSV files.")
    parser.add_argument("--ma-window", type=int, default=0, help="Moving average window length. 0 (default) disables it.")
    parser.add_argument("--subsample-window", type=int, default=0, help="Non-overlapping window size for averaging session rows. 0 (default) disables it.")
    parser.add_argument("--check-gran-empty", action="store_true", help="Truncate session when `gran1_blain` hits 0 or is empty/NaN.")
    parser.add_argument("--chunk-size", type=int, default=0, help="Divide sequences into smaller non-overlapping chunks of this exact size. Smaller leftovers are discarded. 0 (default) disables it.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    all_session_lengths = []
    total_original_rows = 0
    total_processed_rows = 0

    for input_file in csv_files:
        output_file = output_dir / input_file.name
        original_rows, processed_rows, session_lengths = preprocess_file(
            input_file,
            output_file,
            args.ma_window,
            args.subsample_window,
            args.check_gran_empty,
            args.chunk_size
        )

        all_session_lengths.append(session_lengths)
        total_original_rows += original_rows
        total_processed_rows += processed_rows

        print(f"File: {input_file.name}")
        print(f"  Rows Processed: {original_rows} -> {processed_rows}")
        print(f"  Total Sequences: {len(session_lengths)}")
        if not session_lengths.empty:
            print(f"  Sequence Lengths:")
            print(f"    Min: {session_lengths.min()}")
            print(f"    Max: {session_lengths.max()}")
            print(f"    Q1 (25%): {session_lengths.quantile(0.25):.1f}")
            print(f"    Median (50%): {session_lengths.quantile(0.50):.1f}")
            print(f"    Q3 (75%): {session_lengths.quantile(0.75):.1f}")
        print("-" * 40)

    # Print overall statistics after all files are processed
    if all_session_lengths:
        global_lengths = pd.concat(all_session_lengths, ignore_index=True)
        print("=" * 40)
        print("OVERALL STATISTICS")
        print("=" * 40)
        print(f"Total Rows Processed: {total_original_rows} -> {total_processed_rows}")
        print(f"Total Files Processed: {len(csv_files)}")
        print(f"Total Sequences Across All Files: {len(global_lengths)}")
        if not global_lengths.empty:
            print(f"Global Sequence Lengths:")
            print(f"  Min: {global_lengths.min()}")
            print(f"  Max: {global_lengths.max()}")
            print(f"  Q1 (25%): {global_lengths.quantile(0.25):.1f}")
            print(f"  Median (50%): {global_lengths.quantile(0.50):.1f}")
            print(f"  Q3 (75%): {global_lengths.quantile(0.75):.1f}")
        print("=" * 40)


if __name__ == "__main__":
    main()