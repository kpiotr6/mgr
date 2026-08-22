"""Form G -- rank shares across the modelling approaches.

Every matched configuration (model x lookback window x horizon) is scored 1st,
2nd or 3rd by MASE, and each bar shows an approach's share of each place, per
target plus a pooled group over the targets all three approaches cover.

Ranks ignore *how much* an approach won by, so a single diverging run costs one
place rather than dragging a mean. ``gran1_blain`` has no ``all_targets`` run and
is therefore ranked of two; it stays out of the pooled group.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_ranks import (  # noqa: E402
    APPROACH_ORDER, TARGET_ORDER, approaches_for, load_metrics, rank_summary, rank_table,
)

# Ordinal ramp: one hue, dark (1st) to light (3rd), light end still clearing the surface.
PLACE_COLOURS = ["#1c5cab", "#3987e5", "#86b6ef"]
PLACE_LABELS = ["1st place", "2nd place", "3rd place"]

FONT_SIZES = {
    "title": 24.0,
    "group": 19.0,
    "row": 19.0,
    "tick": 18.0,
    "value": 15.0,
    "legend": 18.0,
    "note": 16.0,
}

def format_value(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def build_groups(metrics, targets):
    """One entry per target, plus a pooled entry over the fully-covered ones."""
    groups = []
    for target in targets:
        long = rank_table(metrics, target)
        if long.empty:
            continue
        depth = int(long["Depth"].iloc[0])
        if depth < 2:
            continue
        groups.append({
            "label": target,
            "depth": depth,
            "n": len(long) // depth,
            "summary": rank_summary(long, ["Approach"]).set_index("Approach"),
        })

    full = [t for t in targets if len(approaches_for(metrics, t)) == len(APPROACH_ORDER)]
    if len(full) > 1:
        import pandas as pd
        pooled = pd.concat([rank_table(metrics, t) for t in full], ignore_index=True)
        groups.append({
            "label": f"pooled ({len(full)} targets)",
            "depth": len(APPROACH_ORDER),
            "n": len(pooled) // len(APPROACH_ORDER),
            "summary": rank_summary(pooled, ["Approach"]).set_index("Approach"),
        })
    return groups


def plot_groups(groups, title, output_paths, fonts):
    rows = sum(len(g["summary"]) for g in groups)
    height = 0.66*rows + 0.62*len(groups) + 2.6
    figure, axis = plt.subplots(figsize=(13.0, height))

    # With one group the figure title already names the target; don't repeat it.
    name_groups = len(groups) > 1
    y, ticks, labels = 0.0, [], []
    for group in groups:
        y += 0.9
        header = f"{group['label']}  ·  n = {group['n']}" if name_groups else f"n = {group['n']}"
        axis.text(-0.005, y, header,
                  ha="left", va="center", fontsize=fonts["group"], fontweight="bold",
                  transform=axis.get_yaxis_transform(which="grid"), clip_on=False)
        if group["depth"] < len(APPROACH_ORDER):
            axis.text(1.005, y, f"ranked of {group['depth']}", ha="left", va="center",
                      fontsize=fonts["note"], color="#84847b",
                      transform=axis.get_yaxis_transform(which="grid"), clip_on=False)
        y += 0.85
        for approach in [a for a in APPROACH_ORDER if a in group["summary"].index]:
            row = group["summary"].loc[approach]
            left = 0.0
            for place in range(group["depth"]):
                share = float(row[f"Place{place+1}"])
                if share <= 0:
                    continue
                axis.barh(y, share, left=left, height=0.62,
                          color=PLACE_COLOURS[place], zorder=3)
                if share > 0.07:
                    axis.text(left + share/2, y, f"{round(share*100)}%",
                              ha="center", va="center", fontsize=fonts["value"], zorder=4,
                              color="#16160f" if place == 2 else "#ffffff")
                left += share
            axis.text(1.005, y, f"mean {format_value(float(row['MeanRank']))}",
                      ha="left", va="center", fontsize=fonts["value"], color="#55554e",
                      transform=axis.get_yaxis_transform(which="grid"), clip_on=False)
            ticks.append(y)
            labels.append(approach)
            y += 1.0

    axis.set_yticks(ticks, labels, fontsize=fonts["row"])
    axis.set_ylim(y - 0.5, -0.4)
    axis.set_xlim(0, 1)
    axis.set_xticks(np.arange(0, 1.01, 0.25),
                    [f"{round(v*100)}%" for v in np.arange(0, 1.01, 0.25)],
                    fontsize=fonts["tick"])
    axis.set_xlabel("share of configurations", fontsize=fonts["tick"])
    axis.xaxis.grid(True, color="#e8e8e3", linewidth=1, zorder=0)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)
    axis.set_title(title, fontsize=fonts["title"], pad=62)
    axis.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(PLACE_COLOURS, PLACE_LABELS)],
                loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False,
                fontsize=fonts["legend"])

    figure.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_charts(evaluation_dir, output_dir, formats, font_scale, combined):
    metrics = load_metrics(evaluation_dir)
    fonts = {role: size*font_scale for role, size in FONT_SIZES.items()}
    targets = [t for t in TARGET_ORDER if t in set(metrics["Target"])]

    if not combined:
        for target in targets:
            groups = build_groups(metrics, [target])
            if not groups:
                continue
            title = f"Approach rank shares — {target}"
            paths = [output_dir / f"rank_shares_{target}.{ext}" for ext in formats]
            plot_groups(groups, title, paths, fonts)
            print(f"Saved {paths[0]}")
        return

    groups = build_groups(metrics, targets)
    title = "Approach rank shares by target"
    paths = [output_dir / f"rank_shares_by_target.{ext}" for ext in formats]
    plot_groups(groups, title, paths, fonts)
    print(f"Saved {paths[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot rank shares of the modelling approaches, per target and pooled (Form G)."
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
                        help="Write a single chart covering every target plus a pooled group, "
                             "instead of one chart per target")
    args = parser.parse_args()
    build_charts(Path(args.evaluation_dir), Path(args.output_dir),
                 args.formats, args.font_scale, args.combined)


if __name__ == "__main__":
    main()
