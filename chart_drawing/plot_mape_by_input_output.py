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
    output_lengths: list[int],
    title: str,
    output_path: Path,
    y_axis_max: float | None,
) -> None:
    models = pivot.index.tolist()
    ordered_models = order_models(models)
    pivot = pivot.reindex(ordered_models)
    display_models = [normalize_model_name(name) for name in ordered_models]

    x_positions = np.arange(len(ordered_models))
    bar_width = 0.8 / max(len(output_lengths), 1)

    fig, axis = plt.subplots(figsize=(12, 6))

    for idx, output_length in enumerate(output_lengths):
        values = pivot[output_length].to_numpy()
        offsets = x_positions + (idx - (len(output_lengths) - 1) / 2) * bar_width
        bars = axis.bar(offsets, values, width=bar_width, label=str(output_length))

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
    axis.set_ylabel("MAPE")
    if y_axis_max is not None and y_axis_max > 0:
        axis.set_ylim(0, y_axis_max)
    axis.set_title(title)
    axis.legend(title="OutputChunkLength")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def build_charts(csv_path: Path, output_dir: Path) -> None:
    data_frame = pd.read_csv(csv_path)

    if "InputChunkLength" not in data_frame.columns or "OutputChunkLength" not in data_frame.columns:
        raise ValueError("CSV is missing required columns: InputChunkLength, OutputChunkLength")

    mape_columns = [col for col in data_frame.columns if col.startswith("MAPE_")]

    output_dir.mkdir(parents=True, exist_ok=True)

    input_lengths = sorted(data_frame["InputChunkLength"].unique())

    for input_length in input_lengths:
        input_frame = data_frame[data_frame["InputChunkLength"] == input_length]

        for mape_column in mape_columns:
            metric_frame = input_frame[["Model", "OutputChunkLength", mape_column]].dropna()
            if metric_frame.empty:
                continue

            output_lengths = sorted(metric_frame["OutputChunkLength"].unique())
            pivot = (
                metric_frame.pivot(index="Model", columns="OutputChunkLength", values=mape_column)
                .reindex(sorted(metric_frame["Model"].unique()))
                .reindex(columns=output_lengths)
            )

            metric_name = mape_column.replace("MAPE_", "")
            title = f"MAPE metic value for {metric_name}"
            filename = sanitize_filename(f"mape_{metric_name}_input_{input_length}.png")
            output_path = output_dir / filename

            max_value = float(pivot.to_numpy(np.float64).max())
            y_axis_max = max_value * 1.1 if max_value > 0 else None
            plot_grouped_bars(pivot, output_lengths, title, output_path, y_axis_max)
            print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAPE metrics grouped by input and output chunk length.")
    parser.add_argument(
        "--csv",
        default="/home/kpiotr6/Documents/stuida/praca_magisterska/proj/outputs_ready/evaluation_metrics_all_targets.csv",
        help="Path to evaluation_metrics_all_targets.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/kpiotr6/Documents/stuida/praca_magisterska/proj/chart_drawing/mape_charts",
        help="Directory to save charts",
    )

    args = parser.parse_args()
    build_charts(Path(args.csv), Path(args.output_dir))


if __name__ == "__main__":
    main()
