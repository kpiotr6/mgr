import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    model_order: list[str],
    input_lengths: list[int],
    title: str,
    output_path: Path,
    y_axis_max: float | None,
) -> None:
    ordered_models = order_models(model_order)

    # Reindex the pivot so rows are models (in order) and columns are input lengths
    pivot = pivot.reindex(index=ordered_models, columns=input_lengths)
    display_models = [normalize_model_name(name) for name in ordered_models]

    x_positions = np.arange(len(ordered_models))
    bar_width = 0.8 / max(len(input_lengths), 1)

    fig, axis = plt.subplots(figsize=(12, 6))

    # Iterate over input lengths to draw the bars for each model cluster
    for idx, input_len in enumerate(input_lengths):
        values = pivot[input_len].to_numpy()

        offsets = x_positions + (idx - (len(input_lengths) - 1) / 2) * bar_width

        # Clarify that 1 step = 5 minutes in the legend
        input_minutes = int(input_len) * 5
        label_str = f"{input_len} steps ({input_minutes} min)"

        bars = axis.bar(offsets, values, width=bar_width, label=label_str)

        for bar, value in zip(bars, values, strict=False):
            if np.isnan(value):
                continue

            # Rotate text if there are many bars clustered together to avoid overlap
            text_rotation = 90 if len(input_lengths) >= 4 else 0
            va_alignment = "bottom"

            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                format_value(float(value)),
                ha="center",
                va=va_alignment,
                rotation=text_rotation,
                fontsize=8,
            )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(display_models, rotation=45, ha="right")
    axis.set_xlabel("Model")
    axis.set_ylabel("MAE")

    if y_axis_max is not None and y_axis_max > 0:
        # Increase ceiling slightly more if text is rotated to fit the labels
        axis.set_ylim(0, y_axis_max * (1.15 if len(input_lengths) >= 4 else 1.1))

    axis.set_title(title)

    axis.legend(
        title="Input Chunk Length\n(Step size = 5 min)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_charts(csv_dir: Path, output_dir: Path) -> None:
    target_files = sorted(csv_dir.glob("evaluation_metrics_target_*.csv"))
    if not target_files:
        raise ValueError(f"No target metric files found in {csv_dir} matching evaluation_metrics_target_*.csv")

    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in target_files:
        target_name = csv_path.stem.removeprefix("evaluation_metrics_target_")
        data_frame = pd.read_csv(csv_path)

        required_columns = {"InputChunkLength", "OutputChunkLength", "Model"}
        missing_columns = required_columns - set(data_frame.columns)
        if missing_columns:
            missing_str = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path.name} is missing required columns: {missing_str}")

        metric_column = "MAE"
        if metric_column not in data_frame.columns:
            MAE_columns = [column for column in data_frame.columns if column.startswith("MAE_")]
            if not MAE_columns:
                raise ValueError(f"{csv_path.name} does not contain a MAE column")
            metric_column = MAE_columns[0]

        target_frame = data_frame[["Model", "InputChunkLength", "OutputChunkLength", metric_column]].dropna()
        if target_frame.empty:
            continue

        output_lengths = sorted(target_frame["OutputChunkLength"].dropna().unique())

        for output_length in output_lengths:
            output_frame = target_frame[target_frame["OutputChunkLength"] == output_length]
            if output_frame.empty:
                continue

            input_lengths = sorted(output_frame["InputChunkLength"].dropna().unique())
            if not input_lengths:
                continue

            # Pivot table: Model as index, InputChunkLength as columns
            pivot = output_frame.pivot(index="Model", columns="InputChunkLength", values=metric_column)
            model_order = sorted(output_frame["Model"].dropna().unique())

            # Filter to just order and required columns to be safe before plotting
            pivot = pivot.reindex(model_order).reindex(columns=input_lengths)

            values_array = pivot.to_numpy(dtype=np.float64)
            valid_values = values_array[~np.isnan(values_array)]
            if len(valid_values) == 0:
                continue

            max_value = float(valid_values.max())
            y_axis_max = max_value * 1.1 if max_value > 0 else None

            output_minutes = int(output_length) * 5
            title = (
                f"MAE for {target_name}\n"
                f"Output Chunk: {output_length} steps ({output_minutes} min) | Step size = 5 min"
            )

            filename = f"{sanitize_filename(f'MAE_{target_name}_output_{output_length}')}.png"
            output_path = output_dir / filename

            plot_grouped_bars(
                pivot=pivot,
                model_order=model_order,
                input_lengths=input_lengths,
                title=title,
                output_path=output_path,
                y_axis_max=y_axis_max,
            )
            print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot MAE charts grouped by model, comparing different input lengths for each output length."
    )
    parser.add_argument(
        "--csv-dir",
        default="./outputs_ready",
        help="Directory containing evaluation_metrics_target_*.csv files",
    )
    parser.add_argument(
        "--output-dir",
        default="./chart_drawing/MAE_charts/per_target_by_output_input",
        help="Directory to save charts",
    )

    args = parser.parse_args()
    build_charts(Path(args.csv_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()