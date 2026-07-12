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
    output_lengths: list[int],
    title: str,
    output_path: Path,
    y_axis_max: float | None,
) -> None:
    ordered_models = order_models(model_order)
    pivot = pivot.reindex(ordered_models)
    display_models = [normalize_model_name(name) for name in ordered_models]

    x_positions = np.arange(len(output_lengths))
    bar_width = 0.8 / max(len(ordered_models), 1)

    fig, axis = plt.subplots(figsize=(12, 6))

    for idx, model in enumerate(ordered_models):
        if model not in pivot.index:
            continue

        values = pivot.loc[model, output_lengths].to_numpy()
        offsets = x_positions + (idx - (len(ordered_models) - 1) / 2) * bar_width
        bars = axis.bar(offsets, values, width=bar_width, label=display_models[idx])

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
    axis.set_xticklabels([str(length) for length in output_lengths], rotation=45, ha="right")
    axis.set_xlabel("Output Chunk Length")
    axis.set_ylabel("MAPE")
    if y_axis_max is not None and y_axis_max > 0:
        axis.set_ylim(0, y_axis_max)
    axis.set_title(title)

    axis.legend(
        title="Model",
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

        metric_column = "MAPE"
        if metric_column not in data_frame.columns:
            mape_columns = [column for column in data_frame.columns if column.startswith("MAPE_")]
            if not mape_columns:
                raise ValueError(f"{csv_path.name} does not contain a MAPE column")
            metric_column = mape_columns[0]

        target_frame = data_frame[["Model", "InputChunkLength", "OutputChunkLength", metric_column]].dropna()
        if target_frame.empty:
            continue

        input_lengths = sorted(target_frame["InputChunkLength"].dropna().unique())

        for input_length in input_lengths:
            input_frame = target_frame[target_frame["InputChunkLength"] == input_length]
            if input_frame.empty:
                continue

            output_lengths = sorted(input_frame["OutputChunkLength"].dropna().unique())
            if not output_lengths:
                continue

            pivot = input_frame.pivot(index="Model", columns="OutputChunkLength", values=metric_column)
            model_order = sorted(input_frame["Model"].dropna().unique())
            pivot = pivot.reindex(model_order).reindex(columns=output_lengths)

            values_array = pivot.to_numpy(dtype=np.float64)
            valid_values = values_array[~np.isnan(values_array)]
            if len(valid_values) == 0:
                continue

            max_value = float(valid_values.max())
            y_axis_max = max_value * 1.1 if max_value > 0 else None

            input_minutes = input_length / 6.0
            title = (
                f"MAPE for {target_name}\n"
                f"Input: {input_length} steps ({input_minutes:.1f} min) | Freq: 10s"
            )

            filename = f"{sanitize_filename(f'mape_{target_name}_input_{input_length}')}.png"
            output_path = output_dir / filename

            plot_grouped_bars(
                pivot=pivot,
                model_order=model_order,
                output_lengths=output_lengths,
                title=title,
                output_path=output_path,
                y_axis_max=y_axis_max,
            )
            print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot MAPE charts for each target-specific metrics file by input and output chunk length."
    )
    parser.add_argument(
        "--csv-dir",
        default="./outputs_ready",
        help="Directory containing evaluation_metrics_target_*.csv files",
    )
    parser.add_argument(
        "--output-dir",
        default="./chart_drawing/mape_charts/per_target_by_input_output",
        help="Directory to save charts",
    )

    args = parser.parse_args()
    build_charts(Path(args.csv_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()