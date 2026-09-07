"""MASE heat-maps, models x lookback window length.

One figure per target and forecasting horizon, drawn from the per-target
evaluation files (``evaluation_metrics_target_*.csv``).

The plotted quantity is the mean absolute error scaled by the NaiveLastValue
baseline evaluated on the same target, lookback window and horizon:

    MASE = MAE_model / MAE_naive

1 means "no better than repeating the last value", below 1 is a real gain and
above 1 is worse than the baseline. Pass ``--metric skill`` to plot
``1 - MAE_model / MAE_naive`` instead, centred on 0 with higher being better.

The denominator is the out-of-sample naive error at the matching horizon rather
than the in-sample one-step error of the textbook definition -- the evaluation
CSVs do not carry the latter. The practical effect is that the horizon cancels
out, so grids for different horizons stay directly comparable.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

BASELINE_MODEL = "NaiveLastValue"
STEP_MINUTES = 5.0

# Padding in points between the two-line title and the top of the grid.
TITLE_PAD = 20.0

# Point sizes at --font-scale 1.0. The figures are usually shrunk to a column
# width, so these sit deliberately above the matplotlib defaults.
FONT_SIZES = {
    "title": 25.0,
    "axis_label": 20.0,
    "tick": 20.0,
    "cell": 20.0,
    "colour_bar_label": 20.0,
    "colour_bar_tick": 20.0,
}

# Diverging ramp running worse -> neutral -> better. `BETTER_HIGH_CMAP` suits a
# metric where large is good (skill); MASE is reversed, since low is good.
WORSE_ARM = ["#7c1f1c", "#b8332f", "#de6f6d", "#eda6a5", "#f8dbdb"]
NEUTRAL = "#f0efec"
BETTER_ARM = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
BETTER_HIGH_CMAP = LinearSegmentedColormap.from_list(
    "worse_to_better", WORSE_ARM + [NEUTRAL] + BETTER_ARM
)
BETTER_LOW_CMAP = BETTER_HIGH_CMAP.reversed()


def normalize_model_name(model_name: str) -> str:
    return model_name.replace("NeuralForecast_", "")


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")


def format_value(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def order_models(models: list[str]) -> list[str]:
    priority = ["LinearRegression", "NeuralForecast_XLinear", "XLinear", "Chronos2"]
    priority_lookup = {name: idx for idx, name in enumerate(priority)}

    def sort_key(model: str) -> tuple[int, str]:
        normalized = normalize_model_name(model)
        if model in priority_lookup:
            return (priority_lookup[model], normalized)
        if normalized in priority_lookup:
            return (priority_lookup[normalized], normalized)
        return (len(priority) - 1, normalized)

    ordered = sorted(models, key=sort_key)
    if "Chronos2" in ordered:
        ordered = [name for name in ordered if name != "Chronos2"] + ["Chronos2"]
    return ordered


def load_per_target_metrics(evaluation_dir: Path) -> pd.DataFrame:
    """Return one tidy row per target x model x input length x output length."""
    frames = []
    for path in sorted(evaluation_dir.glob("evaluation_metrics_target_*.csv")):
        frame = pd.read_csv(path)
        target = frame["RunTag"].iloc[0].removeprefix("target_")
        mae_column = f"MAE_{target}"
        if mae_column not in frame.columns:
            raise ValueError(f"{path.name} has no {mae_column} column")

        tidy = frame[["InputChunkLength", "OutputChunkLength", "Model", mae_column]].copy()
        tidy.columns = ["InputChunkLength", "OutputChunkLength", "Model", "MAE"]
        tidy["Target"] = target
        frames.append(tidy)

    if not frames:
        raise FileNotFoundError(f"No evaluation_metrics_target_*.csv under {evaluation_dir}")

    metrics = pd.concat(frames, ignore_index=True)

    keys = ["Target", "InputChunkLength", "OutputChunkLength"]
    baseline = metrics[metrics["Model"] == BASELINE_MODEL].set_index(keys)["MAE"]
    metrics["MAE_baseline"] = metrics.set_index(keys).index.map(baseline)
    if metrics["MAE_baseline"].isna().any():
        raise ValueError(f"Missing {BASELINE_MODEL} baseline for some configurations")

    metrics["Skill"] = 1.0 - metrics["MAE"] / metrics["MAE_baseline"]
    metrics["Mase"] = metrics["MAE"] / metrics["MAE_baseline"]
    return metrics[metrics["Model"] != BASELINE_MODEL].reset_index(drop=True)


def _round_up(limit: float) -> float:
    return max(np.ceil(limit * 20.0) / 20.0, 0.1)


def colour_limits(values: np.ndarray, centre: float, mode: str) -> tuple[float, float]:
    """Colour-scale reach below and above `centre`, as positive distances.

    ``robust`` and ``full`` keep the two arms equal, so a blue and a red of the
    same intensity mean the same magnitude. ``per-arm`` scales each side on its
    own 95th percentile, which recovers detail on the short side when the data
    is lopsided -- at the cost of that equal-intensity reading.
    """
    deviations = values - centre
    if mode == "full":
        limit = _round_up(float(np.abs(deviations).max()))
        return limit, limit
    if mode == "per-arm":
        below = deviations[deviations < 0]
        above = deviations[deviations > 0]
        low = _round_up(float(np.quantile(np.abs(below), 0.95))) if below.size else 0.1
        high = _round_up(float(np.quantile(above, 0.95))) if above.size else 0.1
        return low, high
    limit = _round_up(float(np.quantile(np.abs(deviations), 0.95)))
    return limit, limit


def plot_heatmap(
    pivot: pd.DataFrame,
    title: str,
    colour_label: str,
    centre: float,
    limits: tuple[float, float],
    colour_map: LinearSegmentedColormap,
    fonts: dict[str, float],
    output_paths: list[Path],
) -> int:
    """Draw one models x input-length heat-map. Returns the clipped cell count."""
    low_limit, high_limit = limits
    ordered_models = order_models(pivot.index.tolist())
    pivot = pivot.reindex(ordered_models)
    display_models = [normalize_model_name(name) for name in ordered_models]
    input_lengths = list(pivot.columns)
    values = pivot.to_numpy(dtype=np.float64)

    finite = values[np.isfinite(values)]
    below_count = int(np.sum(finite < centre - low_limit))
    above_count = int(np.sum(finite > centre + high_limit))
    extend = ("both" if below_count and above_count
              else "min" if below_count
              else "max" if above_count
              else "neither")

    figure, axis = plt.subplots(
        figsize=(1.5 * len(input_lengths) + 4.2, 0.66 * len(ordered_models) + 3.0)
    )
    norm = TwoSlopeNorm(vmin=centre - low_limit, vcenter=centre, vmax=centre + high_limit)
    mesh = axis.imshow(values, cmap=colour_map, norm=norm, aspect="auto")

    axis.set_xticks(np.arange(len(input_lengths)), [str(length) for length in input_lengths])
    axis.set_yticks(np.arange(len(ordered_models)), display_models)
    axis.tick_params(axis="both", labelsize=fonts["tick"])
    axis.set_xlabel("Lookback window length", fontsize=fonts["axis_label"])
    axis.set_title(title, fontsize=fonts["title"], pad=TITLE_PAD)

    # Hairline separators instead of drawn cell borders.
    axis.set_xticks(np.arange(len(input_lengths) + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(len(ordered_models) + 1) - 0.5, minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isnan(value):
                continue
            arm = high_limit if value >= centre else low_limit
            deviation = abs(value - centre) / arm
            text_colour = "white" if deviation > 0.55 else "#16160f"
            axis.text(
                column,
                row,
                format_value(value),
                ha="center",
                va="center",
                fontsize=fonts["cell"],
                color=text_colour,
            )

    colour_bar = figure.colorbar(mesh, ax=axis, fraction=0.035, pad=0.03, extend=extend)
    colour_bar.set_label(colour_label, fontsize=fonts["colour_bar_label"])
    colour_bar.ax.tick_params(labelsize=fonts["colour_bar_tick"])
    colour_bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}".replace(".", ",")))
    colour_bar.outline.set_visible(False)

    figure.tight_layout()
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return below_count + above_count


def build_charts(
    evaluation_dir: Path,
    output_dir: Path,
    metric: str,
    vlim_mode: str,
    formats: list[str],
    font_scale: float,
) -> None:
    metrics = load_per_target_metrics(evaluation_dir)
    fonts = {role: size * font_scale for role, size in FONT_SIZES.items()}
    if metric == "mase":
        column, centre, colour_map = "Mase", 1.0, BETTER_LOW_CMAP
    else:
        column, centre, colour_map = "Skill", 0.0, BETTER_HIGH_CMAP
    metric_label = "MASE" if metric == "mase" else "MAE skill"

    # One colour scale for every chart, so targets as well as horizons stay
    # directly comparable -- the same colour means the same value everywhere.
    limits = colour_limits(metrics[column].to_numpy(dtype=np.float64), centre, vlim_mode)

    for target, target_frame in metrics.groupby("Target", sort=True):
        for output_length, cell_frame in target_frame.groupby("OutputChunkLength", sort=True):
            pivot = cell_frame.pivot_table(
                index="Model", columns="InputChunkLength", values=column, aggfunc="mean"
            )
            output_minutes = output_length * STEP_MINUTES
            title = (
                f"{metric_label} — {target} (per_target)\n"
                f"Forecasting horizon: {output_length} step{'s' if output_length != 1 else ''}"
                f" ({output_minutes:.0f} min) | Freq: 5min"
            )
            stem = sanitize_filename(
                f"{metric_label.lower()}_heatmap_per_target_{target}_horizon_{output_length}"
            )
            paths = [output_dir / f"{stem}.{extension}" for extension in formats]

            clipped = plot_heatmap(pivot, title, metric_label, centre, limits, colour_map, fonts, paths)
            span = f"−{limits[0]:.2f}/+{limits[1]:.2f}"
            note = f" ({clipped} cell(s) outside {span}, value still printed)" if clipped else ""
            print(f"Saved {paths[0]}{note}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot MASE heat-maps (models x lookback window length) per target and forecasting horizon."
    )
    parser.add_argument(
        "--evaluation-dir",
        default="./outputs_ready",
        help="Directory holding evaluation_metrics_target_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="./chart_drawing/mase_heatmaps",
        help="Directory to save charts",
    )
    parser.add_argument(
        "--metric",
        choices=["mase", "skill"],
        default="mase",
        help="mase = MAE/MAE_naive, centred on 1 (default); skill = 1 - MAE/MAE_naive, centred on 0",
    )
    parser.add_argument(
        "--vlim-mode",
        choices=["robust", "full", "per-arm"],
        default="robust",
        help=(
            "robust (default) clips both arms at the 95th percentile; full spans every value; "
            "per-arm scales each arm independently, recovering detail on lopsided targets"
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Output formats to write for each chart (e.g. png pdf svg)",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to every font size (1.25 for a further 25%% bump)",
    )

    args = parser.parse_args()
    build_charts(
        Path(args.evaluation_dir),
        Path(args.output_dir),
        args.metric,
        args.vlim_mode,
        args.formats,
        args.font_scale,
    )


if __name__ == "__main__":
    main()
