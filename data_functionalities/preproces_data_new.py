from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import INPUT_COLS, PAST_COLS, RAW_TIME_COL, TARGET_COLS, TIME_COL


EDGE_COL = "is_at_edge"
GRAN1_BLAIN_COL = "gran1_blain"
IMPUTE_MIN_GAP = pd.Timedelta(seconds=10)
IMPUTE_MAX_GAP = pd.Timedelta(hours=1)
IMPUTE_STEP = pd.Timedelta(seconds=1)


def sort_key(path: Path) -> tuple[int, str]:
	digits = "".join(character for character in path.stem if character.isdigit())
	if digits:
		return int(digits), path.stem
	return 10**9, path.stem


def get_target_columns() -> list[str]:
	columns: list[str] = []
	for column in INPUT_COLS + PAST_COLS + TARGET_COLS:
		if column not in columns:
			columns.append(column)
	return columns


def replace_zeros_with_last_non_zero(series: pd.Series) -> pd.Series:
	numeric_series = pd.to_numeric(series, errors="coerce")
	numeric_series = numeric_series.mask(numeric_series == 0)
	return numeric_series.ffill()


def preprocess_single_file(file_path: Path, output_path: Path, data_columns: list[str]) -> None:
	header = pd.read_csv(file_path, nrows=0)
	available_columns = set(header.columns)

	if RAW_TIME_COL not in available_columns:
		raise ValueError(f"File '{file_path.name}' is missing required column '{RAW_TIME_COL}'.")

	use_columns = [RAW_TIME_COL] + [column for column in data_columns if column in available_columns]
	data = pd.read_csv(file_path, usecols=use_columns)

	for column in data_columns:
		if column not in data.columns:
			data[column] = pd.NA

	data[RAW_TIME_COL] = pd.to_datetime(data[RAW_TIME_COL], errors="coerce")
	data = data.dropna(subset=[RAW_TIME_COL]).copy()
	data = data.sort_values(by=RAW_TIME_COL).reset_index(drop=True)

	if GRAN1_BLAIN_COL in data.columns:
		data[GRAN1_BLAIN_COL] = replace_zeros_with_last_non_zero(data[GRAN1_BLAIN_COL])

	output_rows: list[dict[str, object]] = []
	time_index = 0
	previous_row_values: dict[str, object] | None = None
	previous_timestamp: pd.Timestamp | None = None

	for _, row in data.iterrows():
		current_timestamp = row[RAW_TIME_COL]

		if previous_timestamp is not None and previous_row_values is not None:
			gap = current_timestamp - previous_timestamp

			if IMPUTE_MIN_GAP < gap < IMPUTE_MAX_GAP:
				missing_steps = int(gap.total_seconds()) - 1
				for step_index in range(1, missing_steps + 1):
					imputed_row = {
						TIME_COL: time_index,
						RAW_TIME_COL: previous_timestamp + pd.Timedelta(seconds=step_index),
						EDGE_COL: False,
					}
					imputed_row.update(previous_row_values)
					output_rows.append(imputed_row)
					time_index += 1

		row_values = {column: row[column] for column in data_columns}

		is_edge = previous_timestamp is None
		if previous_timestamp is not None:
			is_edge = (current_timestamp - previous_timestamp) >= IMPUTE_MAX_GAP

		output_row = {
			TIME_COL: time_index,
			RAW_TIME_COL: current_timestamp,
			EDGE_COL: is_edge,
		}
		output_row.update(row_values)
		output_rows.append(output_row)

		time_index += 1
		previous_timestamp = current_timestamp
		previous_row_values = row_values

	ordered_columns = [TIME_COL, RAW_TIME_COL, EDGE_COL] + data_columns
	processed = pd.DataFrame(output_rows, columns=ordered_columns)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	processed.to_csv(output_path, index=False)


def main() -> None:
	parser = argparse.ArgumentParser(description="Preprocess all CSV files from data/ to data_preprocessed/.")
	parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Input directory with CSV files")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("data_preprocessed"),
		help="Output directory for preprocessed CSV files",
	)
	parser.add_argument(
		"--file",
		type=str,
		default=None,
		help="Optional single file to process (example: data3 or data3.csv)",
	)
	args = parser.parse_args()

	csv_files = sorted(args.data_dir.glob("*.csv"), key=sort_key)
	if not csv_files:
		raise FileNotFoundError(f"No CSV files found in {args.data_dir}")

	if args.file:
		selected_name = Path(args.file).name
		allowed_names = {selected_name}
		if not selected_name.endswith(".csv"):
			allowed_names.add(f"{selected_name}.csv")
		csv_files = [file_path for file_path in csv_files if file_path.name in allowed_names]
		if not csv_files:
			raise FileNotFoundError(f"File '{args.file}' not found in {args.data_dir}")

	selected_columns = get_target_columns()

	for file_path in csv_files:
		output_path = args.output_dir / file_path.name
		preprocess_single_file(file_path=file_path, output_path=output_path, data_columns=selected_columns)
		print(f"Saved {output_path}")


if __name__ == "__main__":
	main()
