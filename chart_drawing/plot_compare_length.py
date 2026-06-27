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
    input_lengths: list[int],
    title: str,
    output_path: Path,
    y_axis_max: float | None,
    y_label: str,
) -> None:
    models = pivot.index.tolist()
    ordered_models = order_models(models)
    pivot = pivot.reindex(ordered_models)
    display_models = [normalize_model_name(name) for name in ordered_models]

    x_positions = np.arange(len(ordered_models))
    bar_width = 0.8 / max(len(input_lengths), 1)

    fig, axis = plt.subplots(figsize=(12, 6))

    for idx, input_length in enumerate(input_lengths):
        values = pivot[input_length].to_numpy()
        offsets = x_positions + (idx - (len(input_lengths) - 1) / 2) * bar_width

        # Calculate minutes based on 6 points = 1 minute
        input_minutes = input_length / 6.0
        label_str = f"{input_length} steps ({input_minutes:.1f} min)"

        bars = axis.bar(offsets, values, width=bar_width, label=label_str)

        for bar, value in zip(bars, values, strict=True):
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
    axis.set_ylabel(y_label)  # Fixed to use the dynamic y_label instead of hardcoded "MAPE"
    if y_axis_max is not None and y_axis_max > 0:
        axis.set_ylim(0, y_axis_max)
    axis.set_title(title)

    # Move legend completely outside the plot area to the right
    axis.legend(
        title="Input Chunk Length",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0
    )

    fig.tight_layout()
    # Add bbox_inches="tight" so the external legend isn't chopped off in the saved file
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_charts(csv_path: Path, output_dir: Path) -> None:
    data_frame = pd.read_csv(csv_path)

    if "InputChunkLength" not in data_frame.columns or "OutputChunkLength" not in data_frame.columns:
        raise ValueError("CSV is missing required columns: InputChunkLength, OutputChunkLength")

    metric_prefixes = ["MAPE_", "MAE_", "MSE_"]
    metric_columns = [
        col for col in data_frame.columns if any(col.startswith(prefix) for prefix in metric_prefixes)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over OutputChunkLengths instead of InputChunkLengths
    output_lengths = sorted(data_frame["OutputChunkLength"].unique())

    for output_length in output_lengths:
        output_frame = data_frame[data_frame["OutputChunkLength"] == output_length]

        # Calculate output minutes based on 6 points = 1 minute
        output_minutes = output_length / 6.0

        for metric_column in metric_columns:
            # Pivot around InputChunkLength this time
            metric_frame = output_frame[["Model", "InputChunkLength", metric_column]].dropna()
            if metric_frame.empty:
                continue

            input_lengths = sorted(metric_frame["InputChunkLength"].unique())
            pivot = (
                metric_frame.pivot(index="Model", columns="InputChunkLength", values=metric_column)
                .reindex(sorted(metric_frame["Model"].unique()))
                .reindex(columns=input_lengths)
            )

            metric_name_full = metric_column
            metric_prefix = next(prefix for prefix in metric_prefixes if metric_name_full.startswith(prefix))
            metric_short_name = metric_prefix.replace("_", "")
            metric_name_without_prefix = metric_name_full.replace(metric_prefix, "")

            # Updated title to include time info, output minutes, and frequency cleanly over two lines
            title = (
                f"{metric_short_name} metric value for {metric_name_without_prefix}\n"
                f"Output: {output_length} steps ({output_minutes:.1f} min) | Freq: 10s"
            )

            filename = sanitize_filename(f"{metric_short_name.lower()}_{metric_name_without_prefix}_output_{output_length}.png")
            output_path = output_dir / filename

            max_value = float(pivot.to_numpy(np.float64).max())
            y_axis_max = max_value * 1.1 if max_value > 0 else None

            # Pass input_lengths to the plotting function
            plot_grouped_bars(pivot, input_lengths, title, output_path, y_axis_max, metric_short_name)
            print(f"Saved {output_path}")


def main() -> None:
    # Updated description and default output directory
    parser = argparse.ArgumentParser(description="Plot metrics grouped by input chunk length for fixed output lengths.")
    parser.add_argument(
        "--csv",
        default="./outputs_ready/evaluation_metrics_all_targets.csv",
        help="Path to evaluation_metrics_all_targets.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="./chart_drawing/input_influence_charts",
        help="Directory to save charts",
    )

    args = parser.parse_args()
    build_charts(Path(args.csv), Path(args.output_dir))


if __name__ == "__main__":
    main()