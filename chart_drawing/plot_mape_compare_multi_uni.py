import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize_model_name(model_name: str) -> str:
    return model_name.replace("NeuralForecast_", "")


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")


def format_value(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def order_models(models: list[str]) -> list[str]:
    priority = ["NaiveLastValue", "LinearRegression", "NeuralForecast_XLinear", "XLinear", "Chronos2"]
    priority_lookup = {name: idx for idx, name in enumerate(priority)}

    def sort_key(model: str) -> tuple[int, str]:
        normalized = normalize_model_name(model)
        if model in priority_lookup:
            return (priority_lookup[model], normalized)
        if normalized in priority_lookup:
            return (priority_lookup[normalized], normalized)
        if normalized == "Chronos2":
            return (priority_lookup["Chronos2"], normalized)
        return (len(priority) - 1, normalized)

    ordered = sorted(models, key=sort_key)
    if "Chronos2" in ordered:
        ordered = [name for name in ordered if name != "Chronos2"] + ["Chronos2"]
    return ordered


def plot_grouped_bars(
    pivot: pd.DataFrame,
    groups: list[str],
    title: str,
    output_path: Path,
    y_axis_max: float | None,
) -> None:
    models = pivot.index.tolist()
    ordered_models = order_models(models)
    pivot = pivot.reindex(ordered_models)
    display_models = [normalize_model_name(name) for name in ordered_models]

    x_positions = np.arange(len(ordered_models))
    bar_width = 0.8 / max(len(groups), 1)

    fig, axis = plt.subplots(figsize=(12, 6))

    for idx, group in enumerate(groups):
        # Fallback to avoid KeyError if a group happens to be missing entirely
        if group not in pivot.columns:
            continue

        values = pivot[group].to_numpy()
        offsets = x_positions + (idx - (len(groups) - 1) / 2) * bar_width
        bars = axis.bar(offsets, values, width=bar_width, label=group)

        for bar, value in zip(bars, values, strict=False):
            if np.isnan(value):
                continue
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                format_value(float(value)),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(display_models, rotation=45, ha="right")
    axis.set_xlabel("Model")
    axis.set_ylabel("MAPE")
    if y_axis_max is not None and y_axis_max > 0:
        axis.set_ylim(0, y_axis_max)
    axis.set_title(title)

    # Place legend completely outside the plot area to the right
    axis.legend(
        title="Source",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0
    )

    fig.tight_layout()
    # Add bbox_inches="tight" to ensure the saved PNG expands to fit the legend
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_charts(csv_all_path: Path, csv_dir: Path, output_dir: Path) -> None:
    df_all = pd.read_csv(csv_all_path)

    if "InputChunkLength" not in df_all.columns or "OutputChunkLength" not in df_all.columns:
        raise ValueError("CSV is missing required columns: InputChunkLength, OutputChunkLength")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Automatically find all target files in the directory
    target_files = list(csv_dir.glob("evaluation_metrics_target_*.csv"))

    if not target_files:
        print(f"No target files found in directory {csv_dir} matching pattern 'evaluation_metrics_target_*.csv'")
        return

    for target_csv in target_files:
        # Extract target name from the filename
        target_name = target_csv.name.replace("evaluation_metrics_target_", "").replace(".csv", "")

        df_single = pd.read_csv(target_csv)
        mape_col_all = f"MAPE_{target_name}"

        if mape_col_all not in df_all.columns:
            print(f"Skipping {target_name}, '{mape_col_all}' not found in the all_targets file.")
            continue

        # Determine the metric column in the single target file
        mape_col_single = f"MAPE_{target_name}"
        if mape_col_single not in df_single.columns:
            if 'MAPE' in df_single.columns:
                mape_col_single = 'MAPE'
            else:
                print(f"Skipping {target_name}, metric column not found in the concrete target file.")
                continue

        # Extract and uniformize All Targets dataframe subset
        df_all_target = df_all[['Model', 'InputChunkLength', 'OutputChunkLength', mape_col_all]].copy()
        df_all_target.rename(columns={mape_col_all: 'MAPE_Val'}, inplace=True)
        df_all_target['Source'] = 'All Targets'

        # Extract and uniformize Single Target dataframe subset
        df_single_target = df_single[['Model', 'InputChunkLength', 'OutputChunkLength', mape_col_single]].copy()
        df_single_target.rename(columns={mape_col_single: 'MAPE_Val'}, inplace=True)
        df_single_target['Source'] = 'Single Target'

        # Combine the sets into one vertical dataframe
        df_combined = pd.concat([df_all_target, df_single_target], ignore_index=True).dropna(subset=['MAPE_Val'])
        if df_combined.empty:
            continue

        input_lengths = sorted(df_combined["InputChunkLength"].unique())

        for input_length in input_lengths:
            input_frame = df_combined[df_combined["InputChunkLength"] == input_length]
            output_lengths = sorted(input_frame["OutputChunkLength"].unique())

            # Calculate input minutes (6 points = 1 minute)
            input_minutes = input_length / 6.0

            # Generate a distinct plot for EVERY OutputChunkLength
            for output_length in output_lengths:
                output_frame = input_frame[input_frame["OutputChunkLength"] == output_length]

                # Calculate output minutes (6 points = 1 minute)
                output_minutes = output_length / 6.0

                # The groups are strictly Single vs All now
                groups_ordered = ["All Targets", "Single Target"]

                pivot = (
                    output_frame.pivot(index="Model", columns="Source", values="MAPE_Val")
                    .reindex(sorted(output_frame["Model"].unique()))
                )

                # Filter to only the sources we have data for
                existing_groups = [g for g in groups_ordered if g in pivot.columns]
                pivot = pivot.reindex(columns=existing_groups)

                # Updated Title with converted minutes and frequency
                title = (
                    f"MAPE comparison for {target_name}\n"
                    f"Input: {input_length} steps ({input_minutes:.1f} min), "
                    f"Output: {output_length} steps ({output_minutes:.1f} min) | Freq: 10s"
                )

                filename = sanitize_filename(f"mape_compare_{target_name}_in_{input_length}_out_{output_length}.png")
                output_path = output_dir / filename

                # Calculate upper bound safely (discarding NaNs)
                values_array = pivot.to_numpy(np.float64)
                valid_values = values_array[~np.isnan(values_array)]

                if len(valid_values) == 0:
                    continue

                max_value = float(valid_values.max())
                y_axis_max = max_value * 1.1 if max_value > 0 else None

                plot_grouped_bars(pivot, existing_groups, title, output_path, y_axis_max)
                print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAPE metrics comparison: all targets vs single target models.")
    parser.add_argument(
        "--csv",
        default="./outputs_ready/evaluation_metrics_all_targets.csv",
        help="Path to evaluation_metrics_all_targets.csv",
    )
    parser.add_argument(
        "--csv-dir",
        default="./outputs_ready",
        help="Directory containing the concrete target evaluation metrics files (e.g., evaluation_metrics_target_first_chamber_filling.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="./chart_drawing/mape_charts",
        help="Directory to save charts",
    )

    args = parser.parse_args()
    build_charts(Path(args.csv), Path(args.csv_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()