"""Form H -- does the best approach depend on the model?

One panel per target, one line per model, tracked across the approaches that
target has. The y axis is mean rank by MASE, so the winning end is the bottom of
the panel; the count the ranks come from is carried by the title.

Parallel lines would mean the approach choice is separable from the model;
crossing lines mean it is not. ``gran1_blain`` has no ``all_targets`` run, so its
panel is a two-approach comparison on a mean-rank-of-two scale -- not directly
comparable with the three-approach panels.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_ranks import (  # noqa: E402
    TARGET_ORDER, approaches_for, load_metrics, rank_summary, rank_table,
)

# Categorical hues, assigned to models in a fixed order and never cycled.
MODEL_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]

FONT_SIZES = {
    "suptitle": 25.0,
    "panel": 21.0,
    "subtitle": 15.0,
    "tick": 19.0,
    "axis_label": 19.0,
    "legend": 18.0,
}

def format_value(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def order_models(models: list[str]) -> list[str]:
    priority = ["LinearRegression", "XLinear", "Chronos2"]
    lookup = {name: idx for idx, name in enumerate(priority)}

    def sort_key(model: str) -> tuple[int, str]:
        return (lookup.get(model, len(priority) - 1), model)

    ordered = sorted(models, key=sort_key)
    if "Chronos2" in ordered:
        ordered = [m for m in ordered if m != "Chronos2"] + ["Chronos2"]
    return ordered


def series_and_offsets(summary, models, approaches):
    """Mean-rank series per model, plus nudges that keep tied models visible.

    Models that tie on every approach would plot exactly on top of each other,
    hiding all but the last drawn.
    """
    series = {m: [float(summary.loc[(m, a), "MeanRank"]) for a in approaches] for m in models}

    groups: dict[tuple, list[str]] = {}
    for model, values in series.items():
        groups.setdefault(tuple(round(v, 6) for v in values), []).append(model)
    span = max(max(v) for v in series.values()) - min(min(v) for v in series.values())
    step = (span or 1.0) * 0.009
    offsets = {}
    for members in groups.values():
        for index, model in enumerate(members):
            offsets[model] = (index - (len(members) - 1)/2) * step
    return series, offsets


def panel_note(panel: dict) -> str:
    """The reading instructions that ride along with the title."""
    return f"mean rank of {panel['depth']} · {panel['per']} configs per point"


def panel_data(metrics, target):
    long = rank_table(metrics, target)
    if long.empty:
        return None
    approaches = approaches_for(metrics, target)
    summary = rank_summary(long, ["Model", "Approach"]).set_index(["Model", "Approach"])
    models = order_models(sorted({m for m, _ in summary.index}))
    series, offsets = series_and_offsets(summary, models, approaches)
    return {
        "target": target,
        "approaches": approaches,
        "summary": summary,
        "models": models,
        "series": series,
        "offsets": offsets,
        "depth": len(approaches),
        "per": int(summary.loc[(models[0], approaches[0]), "N"]),
    }


def draw_panel(axis, panel, colours, fonts, show_ylabel, show_title):
    """Draw one target. ``show_title`` only when faceted -- a single panel is
    titled by the figure, note included."""
    approaches = panel["approaches"]
    positions = np.arange(len(approaches), dtype=float)

    series, offsets = panel["series"], panel["offsets"]

    for model in panel["models"]:
        values = [v + offsets[model] for v in series[model]]
        axis.plot(positions, values, color=colours[model], linewidth=2, zorder=3,
                  marker="o", markersize=7, markerfacecolor=colours[model],
                  markeredgecolor="#fcfcfb", markeredgewidth=1.6)

    axis.set_xticks(positions, approaches, fontsize=fonts["tick"])
    axis.set_xlim(-0.35, len(approaches) - 0.65)
    axis.yaxis.set_major_formatter(lambda v, _: format_value(v))
    axis.tick_params(axis="y", labelsize=fonts["tick"])
    axis.yaxis.grid(True, color="#e8e8e3", linewidth=1)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)
    if show_ylabel:
        axis.set_ylabel("mean rank", fontsize=fonts["axis_label"])

    if show_title:
        # Faceted: each panel names its own target, with its own note beneath it --
        # depth and tie handling differ per target, so one shared note would lie.
        axis.set_title(panel["target"], fontsize=fonts["panel"], loc="left",
                       fontweight="bold", pad=38)
        axis.text(0, 1.012, panel_note(panel), transform=axis.transAxes,
                  fontsize=fonts["subtitle"], color="#84847b", va="bottom")


def plot_panels(panels, title, output_paths, fonts, colours):
    count = len(panels)
    cols = 2 if count == 4 else min(count, 3)
    rows = int(np.ceil(count / cols))
    models = order_models(sorted({m for p in panels for m in p["models"]}))

    # Reserve height for the title and however many rows the legend needs.
    legend_cols = min(len(models), 3 if cols == 1 else 4)
    legend_rows = int(np.ceil(len(models) / legend_cols))
    note = panel_note(panels[0]) if count == 1 else None
    header = 0.75 + 0.42*legend_rows + (0.34 if note else 0.0)
    fig_h = 5.7*rows + header
    figure, axes = plt.subplots(rows, cols, figsize=(7.0*cols, fig_h), squeeze=False)

    for index, panel in enumerate(panels):
        r, c = divmod(index, cols)
        draw_panel(axes[r][c], panel, colours, fonts,
                   show_ylabel=(c == 0), show_title=(count > 1))
    for index in range(count, rows*cols):
        r, c = divmod(index, cols)
        axes[r][c].set_visible(False)

    handles = [Line2D([0], [0], color=colours[m], linewidth=2, marker="o",
                      markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=1.6, label=m)
               for m in models]
    legend_y = 0.62 + (0.34 if note else 0.0)
    figure.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, 1 - legend_y/fig_h),
                  ncol=legend_cols, frameon=False, fontsize=fonts["legend"])
    figure.suptitle(title, fontsize=fonts["suptitle"], y=1 - 0.12/fig_h, va="top")
    if note:
        figure.text(0.5, 1 - 0.60/fig_h, note, ha="center", va="top",
                    fontsize=fonts["subtitle"], color="#84847b")
    figure.tight_layout(rect=(0, 0, 1, 1 - header/fig_h))

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_charts(evaluation_dir, output_dir, formats, font_scale, combined):
    metrics = load_metrics(evaluation_dir)
    fonts = {role: size*font_scale for role, size in FONT_SIZES.items()}
    targets = [t for t in TARGET_ORDER if t in set(metrics["Target"])]

    panels = [p for p in (panel_data(metrics, t) for t in targets) if p and p["depth"] > 1]
    if not panels:
        raise ValueError("No target is covered by at least two approaches")

    all_models = order_models(sorted({m for p in panels for m in p["models"]}))
    colours = {m: MODEL_COLOURS[i % len(MODEL_COLOURS)] for i, m in enumerate(all_models)}

    if not combined:
        for panel in panels:
            paths = [output_dir / f"approach_model_interaction_{panel['target']}.{ext}"
                     for ext in formats]
            plot_panels([panel], f"Approach per model — {panel['target']}", paths, fonts, colours)
            print(f"Saved {paths[0]}")
        return

    paths = [output_dir / f"approach_model_interaction.{ext}" for ext in formats]
    plot_panels(panels, "Approach per model", paths, fonts, colours)
    print(f"Saved {paths[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot mean rank per model across the modelling approaches, faceted by target (Form H)."
    )
    parser.add_argument("--evaluation-dir", default="./outputs_ready",
                        help="Directory holding evaluation_metrics_*.csv")
    parser.add_argument("--output-dir", default="./chart_drawing/approach_charts",
                        help="Directory to save charts")
    parser.add_argument("--formats", nargs="+", default=["png"],
                        help="Output formats to write for each chart (e.g. png pdf svg)")
    parser.add_argument("--font-scale", type=float, default=1.0,
                        help="Multiplier applied to every font size")
    parser.add_argument("--combined", action="store_true",
                        help="Write a single faceted chart covering every target, "
                             "instead of one chart per target")
    args = parser.parse_args()
    build_charts(Path(args.evaluation_dir), Path(args.output_dir),
                 args.formats, args.font_scale, args.combined)


if __name__ == "__main__":
    main()
