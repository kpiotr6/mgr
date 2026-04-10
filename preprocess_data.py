from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import DEFAULT_FREQ, INPUT_COLS, TARGET_COLS, TIME_COL


EDGE_COL = "is_at_edge"


def sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = "".join(character for character in stem if character.isdigit())
    if digits:
        return int(digits), stem
    return 10**9, stem


def get_columns_to_keep() -> list[str]:
    columns: list[str] = [TIME_COL]
    for column in INPUT_COLS + TARGET_COLS:
        if column not in columns:
            columns.append(column)
    return columns


def load_file(file_path: Path, columns_to_keep: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(file_path)

    if TIME_COL not in frame.columns:
        raise ValueError(f"Missing required time column '{TIME_COL}' in {file_path}")

    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="coerce")
    frame = frame.dropna(subset=[TIME_COL]).copy()
    frame = frame.sort_values(TIME_COL).reset_index(drop=True)

    missing_columns = [column for column in columns_to_keep if column not in frame.columns]
    for column in missing_columns:
        frame[column] = pd.NA

    selected = frame[columns_to_keep].copy()
    selected["_source_file"] = file_path.name
    return selected


def preprocess_frames(
    all_rows: pd.DataFrame,
    step: pd.Timedelta,
    long_gap_threshold: pd.Timedelta,
    edge_window: pd.Timedelta,
) -> pd.DataFrame:
    data_columns = [column for column in all_rows.columns if column not in [TIME_COL, "_source_file"]]

    records: list[dict[str, object]] = []
    edge_until: pd.Timestamp | None = None

    prev_time: pd.Timestamp | None = None
    prev_values: dict[str, object] | None = None

    for _, row in all_rows.iterrows():
        current_time = row[TIME_COL]
        current_source = row["_source_file"]
        current_values = {column: row[column] for column in data_columns}

        if prev_time is not None and current_time > prev_time:
            delta = current_time - prev_time

            if delta > step:
                if delta < long_gap_threshold:
                    missing_points = int(delta // step) - 1
                    for point_offset in range(1, missing_points + 1):
                        imputed_time = prev_time + point_offset * step
                        is_edge = bool(edge_until is not None and imputed_time < edge_until)
                        imputed_row: dict[str, object] = {
                            TIME_COL: imputed_time,
                            "_source_file": current_source,
                            EDGE_COL: is_edge,
                        }
                        if prev_values is not None:
                            imputed_row.update(prev_values)
                        records.append(imputed_row)
                else:
                    candidate_edge_until = current_time + edge_window
                    if edge_until is None or candidate_edge_until > edge_until:
                        edge_until = candidate_edge_until

        current_is_edge = bool(edge_until is not None and current_time < edge_until)
        current_row: dict[str, object] = {
            TIME_COL: current_time,
            "_source_file": current_source,
            EDGE_COL: current_is_edge,
        }
        current_row.update(current_values)
        records.append(current_row)

        prev_time = current_time
        prev_values = current_values

    processed = pd.DataFrame(records)
    ordered_columns = [TIME_COL, EDGE_COL] + data_columns + ["_source_file"]
    return processed[ordered_columns]


def impute_zero_gran1_blain(frame: pd.DataFrame) -> pd.DataFrame:
    if "gran1_blain" not in frame.columns:
        return frame

    result = frame.copy()
    gran1_blain = pd.to_numeric(result["gran1_blain"], errors="coerce")
    gran1_blain = gran1_blain.mask(gran1_blain == 0)
    result["gran1_blain"] = gran1_blain.ffill()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess all CSV files from a folder: keep selected columns, "
            "impute short time gaps, ignore long gaps, create sequence index and edge flag."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Input CSV directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_preprocessed"),
        help="Output directory for preprocessed CSV files",
    )
    parser.add_argument(
        "--long-gap-threshold",
        type=str,
        default="1h",
        help="Gaps >= this duration are not imputed and trigger edge window",
    )
    parser.add_argument(
        "--edge-window",
        type=str,
        default="1h",
        help="Duration after a long-gap boundary where is_at_edge is True",
    )
    args = parser.parse_args()

    csv_files = sorted(args.data_dir.glob("*.csv"), key=sort_key)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.data_dir}")

    columns_to_keep = get_columns_to_keep()
    loaded_frames = [load_file(file_path, columns_to_keep) for file_path in csv_files]
    merged = pd.concat(loaded_frames, ignore_index=True)
    merged = merged.sort_values(TIME_COL).reset_index(drop=True)
    merged = impute_zero_gran1_blain(merged)

    step = pd.to_timedelta(DEFAULT_FREQ)
    long_gap_threshold = pd.to_timedelta(args.long_gap_threshold)
    edge_window = pd.to_timedelta(args.edge_window)

    processed = preprocess_frames(merged, step, long_gap_threshold, edge_window)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for file_path in csv_files:
        output_file = args.output_dir / file_path.name
        file_frame = processed[processed["_source_file"] == file_path.name].copy()
        file_frame = file_frame.drop(columns=["_source_file"])
        file_frame.to_csv(output_file, index=False)
        print(f"Saved {output_file} ({len(file_frame)} rows)")


if __name__ == "__main__":
    main()
