"""Closed-loop IAE against the MPC forecasting window.

Reads the ``mpc_closed_loop_iae_O*.csv`` files written by
``mpc2/run_mpc_on_simulator.py`` -- one closed-loop run per forecasting window,
where the number after ``O`` is the models' ``output_chunk_length``, i.e. how
many 5-minute steps ahead each Darts model forecasts (O1 = 5 min, O2 = 10 min,
O3 = 15 min) -- and draws one bar chart per target comparing ``iae_full``:

    IAE = integral |y(t) - setpoint| dt over the whole logged run
          [target units x minutes]

Lower is better. The forecasting window is an *ordered* category, so the bars
use a single-hue ordinal ramp (light = shortest window) rather than one colour
per bar; bar length still carries the value, and the lowest bar is called out.

``mpc_closed_loop_iae_naive.csv`` -- the baseline run where the controller
forecasts nothing (``run_mpc_on_simulator.py``: ``IS_NAIVE``) -- is read too. It
is not a forecasting window, so it sits first on every x-axis in a neutral grey
that stays outside the ordinal ramp.

Targets are named exactly as they are in ``config.py`` / the MPC runner
(``first_chamber_filling``, never "1st chamber"). ``TOTAL_weighted`` is not a
target but the runner's weighted sum across them; it gets its own chart so the
"which window wins overall" question has one answer.

Per-target charts each carry their own y-axis because the targets differ by two
orders of magnitude (``return`` ~ 2.5e3 vs ``gran1_blain`` ~ 1.8e5) -- the same
reason ``--combined`` draws small multiples instead of grouped bars on one axis.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One 5-minute row per model step (mpc2/mpc_controller.py: STEP_MINUTES).
STEP_MINUTES = 5.0

FILE_PATTERN = "mpc_closed_loop_iae_*.csv"
HORIZON_RE = re.compile(r"_O(\d+)\.csv$")
NAIVE_RE = re.compile(r"_naive\.csv$", re.IGNORECASE)

# The naive baseline is not a forecasting window; 0 only orders it first.
NAIVE_HORIZON = 0

# Ordinal ramp, one hue, light -> dark with a longer window. Validated as an
# ordinal ramp (monotone lightness, visible step gaps, light end clears the
# surface); the same blue the other charts in this folder use.
# The light end stops at step 250 (2.06:1 on a light surface) - going lighter
# would let the shortest window's bar recede into the page; a fourth window
# would extend the *dark* end instead.
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

# The baseline is a different kind of thing, not a lighter/darker window, so it
# gets a neutral grey instead of a ramp step.
NAIVE_COLOUR = "#9a9a92"

INK = "#16160f"
INK_MUTED = "#55554e"
GRID = "#e8e8e3"

TARGET_ORDER = [
    "return",
    "first_chamber_filling",
    "second_chamber_filling",
    "gran1_blain",
    "TOTAL_weighted",
]

FONT_SIZES = {
    "title": 24.0,
    "axis": 19.0,
    "tick": 18.0,
    "value": 16.0,
    "note": 15.0,
    "panel": 19.0,
}


def format_value(value: float) -> str:
    """Thesis-style number: comma decimal separator, thin-space thousands.

    Large IAE values (1e5) would be unreadable with two decimals, so anything
    from 1000 up is rounded to whole units.
    """
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:.2f}".replace(".", ",")


def horizon_label(steps: int) -> str:
    if steps == NAIVE_HORIZON:
        return "naive\nbaseline"
    return f"{steps}\n{steps * STEP_MINUTES:.0f} min"


def load_runs(iae_dir: Path) -> pd.DataFrame:
    """Long frame (target, horizon, <metric columns>) over every run file found.

    ``horizon`` is the O-number, or NAIVE_HORIZON for the baseline run.
    """
    paths = sorted(iae_dir.glob(FILE_PATTERN))
    if not paths:
        raise SystemExit(f"No {FILE_PATTERN} files found in {iae_dir}")

    frames = []
    for path in paths:
        match = HORIZON_RE.search(path.name)
        if match:
            horizon = int(match.group(1))
        elif NAIVE_RE.search(path.name):
            horizon = NAIVE_HORIZON
        else:
            print(f"Skipping {path.name}: no _O<n>.csv or _naive.csv suffix", file=sys.stderr)
            continue
        frame = pd.read_csv(path)
        frame["horizon"] = horizon
        frame["source"] = path.name
        frames.append(frame)

    if not frames:
        raise SystemExit(f"No usable run files in {iae_dir} (need _O<n>.csv or _naive.csv)")

    runs = pd.concat(frames, ignore_index=True).sort_values(["target", "horizon"])
    duplicated = runs.duplicated(["target", "horizon"])
    if duplicated.any():
        raise SystemExit(f"Two files claim the same horizon: {sorted(set(runs.loc[duplicated, 'source']))}")
    return runs


def targets_in(runs: pd.DataFrame) -> list[str]:
    """Known targets first, in TARGET_ORDER, then anything else the files hold."""
    present = set(runs["target"])
    ordered = [t for t in TARGET_ORDER if t in present]
    return ordered + sorted(present - set(ordered))


def draw_bars(axis, sub: pd.DataFrame, metric: str, fonts: dict, colours: dict) -> None:
    """One target's bars on `axis`, value-labelled, lowest bar called out."""
    values = sub[metric].to_numpy(dtype=float)
    positions = np.arange(len(sub))
    best = int(np.argmin(values))

    axis.bar(positions, values, width=0.62,
             color=[colours[h] for h in sub["horizon"]], zorder=3)

    headroom = values.max() * 0.16 if values.max() > 0 else 1.0
    for i, value in enumerate(values):
        axis.text(i, value + headroom * 0.13, format_value(value),
                  ha="center", va="bottom", fontsize=fonts["value"],
                  fontweight="bold" if i == best else "normal", color=INK, zorder=4)
    axis.text(best, values[best] + headroom * 0.62, "lowest",
              ha="center", va="bottom", fontsize=fonts["note"], color=INK_MUTED, zorder=4)

    axis.set_xticks(positions, [horizon_label(h) for h in sub["horizon"]],
                    fontsize=fonts["tick"])
    axis.set_ylim(0, values.max() + headroom)  # bars must start at zero
    axis.tick_params(axis="y", labelsize=fonts["tick"])
    axis.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)


def plot_target(runs, target, metric, output_paths, fonts, colours) -> None:
    sub = runs[runs["target"] == target].sort_values("horizon")
    figure, axis = plt.subplots(figsize=(9.6, 6.8))

    draw_bars(axis, sub, metric, fonts, colours)
    axis.set_ylabel("IAE", fontsize=fonts["axis"])
    axis.set_xlabel("forecasting window", fontsize=fonts["axis"])
    axis.set_title(f"Closed-loop IAE — {target}", fontsize=fonts["title"], pad=22)

    figure.tight_layout()
    save(figure, output_paths)


def plot_combined(runs, targets, metric, output_paths, fonts, colours) -> None:
    """Small multiples -- one panel per target, each with its own y-axis.

    Grouping every target onto one axis would be a dual-scale chart in
    disguise: gran1_blain's IAE is ~20x the others and would flatten them.
    """
    columns = min(3, len(targets))
    rows = int(np.ceil(len(targets) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 5.8 * rows))
    flat = np.atleast_1d(axes).ravel()

    for axis, target in zip(flat, targets):
        sub = runs[runs["target"] == target].sort_values("horizon")
        draw_bars(axis, sub, metric, fonts, colours)
        axis.set_title(target, fontsize=fonts["panel"], pad=14)
        axis.set_ylabel("IAE", fontsize=fonts["note"], color=INK_MUTED)
    for axis in flat[len(targets):]:
        axis.set_visible(False)

    figure.suptitle("Closed-loop IAE by forecasting window",
                    fontsize=fonts["title"], y=0.995)
    figure.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.4)
    save(figure, output_paths)


def save(figure, output_paths) -> None:
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_paths[0]}")


def build_charts(iae_dir, output_dir, metric, formats, font_scale, combined) -> None:
    runs = load_runs(Path(iae_dir))
    if metric not in runs.columns:
        raise SystemExit(f"No column '{metric}' in the IAE files. Available: "
                         f"{[c for c in runs.columns if c.startswith('iae')]}")

    fonts = {role: size * font_scale for role, size in FONT_SIZES.items()}
    horizons = sorted(set(runs["horizon"]))
    out_of_ramp = [h for h in horizons if h != NAIVE_HORIZON and not 1 <= h <= len(RAMP)]
    if out_of_ramp:
        raise SystemExit(f"Horizon(s) {out_of_ramp} have no ordinal ramp step "
                         f"(the ramp defines O1..O{len(RAMP)}).")
    # Keyed by the O-number itself, not by position, so O1 is the same blue in
    # every chart no matter which subset of the files is present.
    colours = {h: NAIVE_COLOUR if h == NAIVE_HORIZON else RAMP[h - 1] for h in horizons}
    targets = targets_in(runs)

    if combined:
        paths = [Path(output_dir) / f"mpc_{metric}_by_horizon.{ext}" for ext in formats]
        plot_combined(runs, targets, metric, paths, fonts, colours)
        return

    for target in targets:
        paths = [Path(output_dir) / f"mpc_{metric}_by_horizon_{target}.{ext}" for ext in formats]
        plot_target(runs, target, metric, paths, fonts, colours)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot closed-loop IAE per target against the MPC forecasting window (O1/O2/O3)."
    )
    parser.add_argument("--iae-dir", default="./outputs_ready",
                        help=f"Directory holding {FILE_PATTERN}")
    parser.add_argument("--output-dir", default="./chart_drawing/mpc_iae_charts",
                        help="Directory to save charts")
    parser.add_argument("--metric", default="iae_full",
                        help="Which column to plot (iae_full, iae_controlled, iae_per_minute_full, ...)")
    parser.add_argument("--formats", nargs="+", default=["png"],
                        help="Output formats to write for each chart (e.g. png pdf svg)")
    parser.add_argument("--font-scale", type=float, default=1.0,
                        help="Multiplier applied to every font size")
    parser.add_argument("--combined", action="store_true",
                        help="Write one small-multiples chart covering every target, "
                             "instead of one chart per target")
    args = parser.parse_args()
    build_charts(args.iae_dir, args.output_dir, args.metric,
                 args.formats, args.font_scale, args.combined)


if __name__ == "__main__":
    main()
