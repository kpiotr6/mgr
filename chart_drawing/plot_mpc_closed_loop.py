"""Closed-loop MPC trajectories, laid out in two columns.

Reads the ``mpc_state_per_minute_O*.csv`` files written by
``mpc2/run_mpc_on_simulator.py`` -- one row per simulated minute of a closed-loop
run, one file per forecasting window, where the number after ``O`` is the models'
``output_chunk_length``, i.e. how many 5-minute steps ahead each Darts model
forecasts (1 = 5 min, 2 = 10 min, 3 = 15 min). Every such file found in
``outputs_ready`` gets its own figure, so the three runs can be compared
side by side.

Each figure draws the same six panels as ``mpc2/fig_mpc_closed_loop.png``, but on
a two-column grid instead of one tall column, so it fits a portrait page
without shrinking the traces:

    return                | first_chamber_filling
    second_chamber_filling| gran1_blain
    separator_speed       | fresh_feed_setpoint

The two manipulated variables always occupy the bottom row and are drawn as
steps (they are held constant across each minute, not interpolated) in their own
hue, so "what the controller did" is separable at a glance from "what the plant
did". Each target panel carries its own y-axis -- the targets differ by two
orders of magnitude (``first_chamber_filling`` ~ 50 % vs ``gran1_blain`` ~ 5500
cm2/g) and sharing one axis would flatten all but the largest.

Panels are titled with the *exact* variable names used in ``config.py`` and the
MPC runner (``first_chamber_filling``, never "1st chamber"), so the figure and
the tables can be cross-read without a translation step; the y-label carries
only the unit.

Usage:
    .venv/bin/python chart_drawing/plot_mpc_closed_loop.py
    .venv/bin/python chart_drawing/plot_mpc_closed_loop.py --state-csv mpc2/mpc_state_per_minute_O2.csv
    .venv/bin/python chart_drawing/plot_mpc_closed_loop.py --state-dir mpc2
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_STATE_DIR = "./outputs_ready"
STEM_PREFIX = "mpc_state_per_minute"
FILE_PATTERN = f"{STEM_PREFIX}_O*.csv"
HORIZON_RE = re.compile(r"_O(\d+)\.csv$")

# One 5-minute row per model step (mpc2/mpc_controller.py: STEP_MINUTES).
STEP_MINUTES = 5.0

DEFAULT_TITLE = "MPC controller closed-loop"

# Controlled variables (top rows), in reading order, with their units.
TARGET_PANELS = [
    ("return", "t/h"),
    ("first_chamber_filling", "%"),
    ("second_chamber_filling", "%"),
    ("gran1_blain", "cm$^2$/g"),
]

# Manipulated variables -- always the bottom row.
CONTROL_PANELS = [
    ("separator_speed", "rpm"),
    ("fresh_feed_setpoint", "t/h"),
]

COLUMNS = 2

# Four categorical hues, validated (light surface #fcfcfb): every adjacent pair
# clears CVD dE 8 and the normal-vision floor of 15. Measured trajectory and
# setpoint reference must stay separable *within* a panel; the two control hues
# only have to separate from those two.
MEASURED = "#1c5cab"
SETPOINT = "#b8332f"
CONTROL_COLOURS = {
    "separator_speed": "#6d4aa7",
    "fresh_feed_setpoint": "#c06000",
}
CONTROL_FALLBACK = "#6d4aa7"

INK = "#16160f"
INK_MUTED = "#55554e"
GRID = "#e8e8e3"
SEED_BAND = "#f0efec"
SURFACE = "#fcfcfb"

# Rough footprint of the "setpoint <value>" label, as a fraction of the panel,
# at font scale 1. Used to slide the label to a clear patch (see _label_slot);
# deliberately generous, so a label placed by these numbers still clears the
# trace after matplotlib lays out the real glyphs.
LABEL_WIDTH_FRAC = 0.26
LABEL_HEIGHT_FRAC = 0.10

FONT_SIZES = {
    "title": 24.0,
    "subtitle": 16.0,
    "panel": 19.0,
    "axis": 17.0,
    "tick": 16.0,
    "note": 14.0,
}


def format_value(value: float) -> str:
    """Thesis-style number: comma decimal separator, thin-space thousands."""
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", " ")
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return (text or "0").replace(".", ",")


def load_state(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [c for c in ("t_min", "phase") if c not in frame.columns]
    if missing:
        raise SystemExit(f"{path}: missing column(s) {missing} - is this a "
                         f"{STEM_PREFIX}*.csv written by run_mpc_on_simulator.py?")
    return frame.sort_values("t_min").reset_index(drop=True)


def seed_end(frame: pd.DataFrame) -> float | None:
    """Time at which MPC took over: the last minute still run open-loop.

    ``None`` when the file holds no seeding phase at all (nothing to mark).
    """
    seeding = frame.loc[frame["phase"] == "seeding", "t_min"]
    return float(seeding.max()) if len(seeding) else None


def panels_in(frame: pd.DataFrame) -> tuple[list, list]:
    targets = [(k, u) for k, u in TARGET_PANELS if k in frame.columns]
    controls = [(k, u) for k, u in CONTROL_PANELS if k in frame.columns]
    if not targets and not controls:
        raise SystemExit("No known target or control column found in the state file.")
    return targets, controls


def style_axis(axis, key, unit, fonts) -> None:
    axis.set_title(key, loc="left", fontsize=fonts["panel"], color=INK, pad=10)
    axis.set_ylabel(unit, fontsize=fonts["axis"], color=INK_MUTED)
    axis.tick_params(labelsize=fonts["tick"], length=0, colors=INK_MUTED)
    axis.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)


def mark_seed(axis, t_start, t_seed) -> None:
    """Shade the open-loop seeding phase and mark where MPC takes over.

    Explained once in the figure subtitle rather than annotated per panel: the
    band is only ~30 of ~230 minutes wide, so an in-panel label either does not
    fit or lands on the trace.
    """
    if t_seed is None:
        return
    axis.axvspan(t_start, t_seed, color=SEED_BAND, zorder=0)
    axis.axvline(t_seed, color=INK_MUTED, ls=":", lw=1.4, zorder=2)


def _y_limits(values, setpoint) -> tuple[float, float]:
    """Panel y-range: the trace plus its setpoint line, with 8 % padding.

    Set explicitly (rather than left to autoscale) because the setpoint label
    is placed by arithmetic on these limits - the placement must be done
    against the range that actually gets drawn.
    """
    low = float(min(values.min(), setpoint))
    high = float(max(values.max(), setpoint))
    span = high - low or max(abs(high), 1.0)
    return low - 0.08 * span, high + 0.08 * span


def _label_slot(times, values, setpoint, y_low, y_high, size_scale=1.0, positions=41):
    """Where to hang the setpoint label so it never lands on the trace.

    The label is treated as the box it actually is, not as a point: roughly
    `LABEL_WIDTH_FRAC` of the panel wide and `LABEL_HEIGHT_FRAC` tall (scaled
    with the font), sitting against the setpoint line either above or below.
    That box is slid across the panel on both sides of the line, and the
    position covering the fewest samples wins - ties broken by the distance to
    the nearest sample, then leftmost. Sides with less room than the box needs
    (the panel edge is right there) are not considered at all.

    Measuring the box instead of a single anchor is what stops the label from
    starting in a clear stretch and running into the trace a few minutes later.

    Returns the x centre of the label and its vertical alignment; the box is
    kept a half-width away from both panel edges, so centring it never pushes
    the text over the axis onto the tick labels.
    """
    t_low, t_high = float(times.min()), float(times.max())
    width = LABEL_WIDTH_FRAC * size_scale * (t_high - t_low)
    height = LABEL_HEIGHT_FRAC * size_scale * (y_high - y_low)
    span = y_high - y_low

    best = None
    for centre in np.linspace(t_low + width / 2, t_high - width / 2, positions):
        chunk = values[(times >= centre - width / 2) & (times <= centre + width / 2)]
        above, below = chunk[chunk > setpoint], chunk[chunk <= setpoint]
        for side, band, gap, room in (
            ("above", (setpoint, setpoint + height),
             (above.min() - setpoint) if len(above) else span, y_high - setpoint),
            ("below", (setpoint - height, setpoint),
             (setpoint - below.max()) if len(below) else span, setpoint - y_low),
        ):
            if room < height:
                continue
            covered = int(((chunk >= band[0]) & (chunk <= band[1])).sum())
            score = (-covered, min(gap, span))
            if best is None or score > best[0]:
                best = (score, centre, side)

    if best is None:   # no side has room for the box - fall back to the line's left end
        return t_low + width / 2, "bottom"
    _, centre, side = best
    return centre, "bottom" if side == "above" else "top"


def draw_target(axis, frame, key, unit, fonts) -> None:
    times = frame["t_min"].to_numpy(dtype=float)
    values = frame[key].to_numpy(dtype=float)
    axis.plot(times, values, color=MEASURED, lw=2.0, zorder=3)

    setpoint_col = f"{key}_setpoint"
    if setpoint_col in frame.columns:
        setpoint = float(frame[setpoint_col].iloc[-1])
        axis.axhline(setpoint, color=SETPOINT, ls="--", lw=2.0, zorder=2)
        y_low, y_high = _y_limits(values, setpoint)
        axis.set_ylim(y_low, y_high)
        # Direct-labelled rather than put in a legend: it is one reference line
        # per panel, and the label can carry the value as well as the identity.
        x, va = _label_slot(times, values, setpoint, y_low, y_high,
                            size_scale=fonts["note"] / FONT_SIZES["note"])
        axis.annotate(f"setpoint {format_value(setpoint)}", xy=(x, setpoint),
                      xytext=(0, 3 if va == "bottom" else -3),
                      textcoords="offset points", ha="center", va=va,
                      fontsize=fonts["note"], color=SETPOINT, zorder=4,
                      # Surface-coloured halo, in case a trace still passes
                      # close: the label has to stay readable.
                      bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5, alpha=0.85))
    style_axis(axis, key, unit, fonts)


def draw_control(axis, frame, key, unit, fonts) -> None:
    # where="post": the control chosen at minute t is held for the whole of the
    # following minute, so the trace must be a staircase, not a ramp.
    axis.step(frame["t_min"], frame[key], where="post",
              color=CONTROL_COLOURS.get(key, CONTROL_FALLBACK), lw=2.0, zorder=3)
    style_axis(axis, key, unit, fonts)


def build_figure(frame: pd.DataFrame, fonts: dict, title: str | None):
    targets, controls = panels_in(frame)
    rows = math.ceil(len(targets) / COLUMNS) + math.ceil(len(controls) / COLUMNS)

    figure, axes = plt.subplots(rows, COLUMNS, figsize=(7.6 * COLUMNS, 3.9 * rows),
                                sharex=True, squeeze=False)
    flat = list(axes.ravel())
    control_start = math.ceil(len(targets) / COLUMNS) * COLUMNS  # controls always start a new row
    used = []

    for axis, (key, unit) in zip(flat, targets):
        draw_target(axis, frame, key, unit, fonts)
        used.append(axis)
    for axis, (key, unit) in zip(flat[control_start:], controls):
        draw_control(axis, frame, key, unit, fonts)
        used.append(axis)
    for axis in flat:
        if axis not in used:
            axis.set_visible(False)

    t_seed = seed_end(frame)
    t_start, t_end = float(frame["t_min"].min()), float(frame["t_min"].max())
    for axis in used:
        # Set the limits before shading, so the band cannot spill into the
        # margin matplotlib would otherwise autoscale in around it.
        axis.set_xlim(t_start, t_end)
        mark_seed(axis, t_start, t_seed)

    # Only the panels with nothing below them carry the x-label; sharex hides
    # the tick labels of the rest anyway.
    for axis in used[-COLUMNS:]:
        axis.set_xlabel("time (min)", fontsize=fonts["axis"], color=INK_MUTED)

    if title:
        figure.suptitle(title, fontsize=fonts["title"], color=INK, y=0.995)
    subtitle = ("a new control is chosen every simulated minute; "
                "the models still see 5-minute mean-pooled history")
    if t_seed is not None:
        subtitle += "\nshaded: open-loop seeding at nominal control, before MPC takes over"
    figure.text(0.5, 0.965 if title else 0.99, subtitle, ha="center", va="top",
                fontsize=fonts["subtitle"], color=INK_MUTED)

    figure.tight_layout(rect=[0, 0, 1, 0.93 if title else 0.955], h_pad=3.0, w_pad=3.0)
    return figure


def save(figure, output_paths) -> None:
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_paths[0]}")


def run_suffix(path: Path) -> str:
    """``mpc_state_per_minute_O2.csv`` -> ``_O2`` (``''`` for the plain file)."""
    stem = path.stem
    return stem[len(STEM_PREFIX):] if stem.startswith(STEM_PREFIX) else f"_{stem}"


def horizon_of(path: Path) -> int | None:
    """The models' ``output_chunk_length`` for this run, from its file name."""
    match = HORIZON_RE.search(path.name)
    return int(match.group(1)) if match else None


def figure_title(base: str, path: Path) -> str | None:
    """Base title, extended with the run's forecasting window when known.

    The window is named by its step count *and* its wall-clock length -- "2"
    alone means nothing to a reader who does not know the models' step size.
    """
    if not base:
        return None
    steps = horizon_of(path)
    if steps is None:
        return base
    return f"{base} \u2014 forecasting window {steps} ({steps * STEP_MINUTES:.0f} min)"


def resolve_inputs(state_csvs, state_dir) -> list[Path]:
    """Explicit ``--state-csv`` files if given, else every run in ``state_dir``.

    Sorted by horizon rather than by name, so a two-digit window would still
    come after the single-digit ones.
    """
    if state_csvs:
        return [Path(p) for p in state_csvs]
    paths = sorted(Path(state_dir).glob(FILE_PATTERN),
                   key=lambda p: (horizon_of(p) is None, horizon_of(p) or 0, p.name))
    if not paths:
        raise SystemExit(f"No {FILE_PATTERN} files found in {state_dir}")
    return paths


def build_charts(state_csvs, state_dir, output_dir, formats, font_scale, title) -> None:
    fonts = {role: size * font_scale for role, size in FONT_SIZES.items()}
    for path in resolve_inputs(state_csvs, state_dir):
        frame = load_state(path)
        figure = build_figure(frame, fonts, figure_title(title, path))
        paths = [Path(output_dir) / f"fig_mpc_closed_loop{run_suffix(path)}.{ext}"
                 for ext in formats]
        save(figure, paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot MPC closed-loop trajectories on a two-column grid, "
                    "with separator_speed and fresh_feed_setpoint on the bottom row."
    )
    parser.add_argument("--state-csv", nargs="+", default=None,
                        help=f"One or more {STEM_PREFIX}*.csv files (one figure each); "
                             f"defaults to every {FILE_PATTERN} in --state-dir")
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR,
                        help=f"Directory holding {FILE_PATTERN} (used when "
                             "--state-csv is not given)")
    parser.add_argument("--output-dir", default="./chart_drawing/mpc_closed_loop_charts",
                        help="Directory to save charts")
    parser.add_argument("--formats", nargs="+", default=["png"],
                        help="Output formats to write for each chart (e.g. png pdf svg)")
    parser.add_argument("--font-scale", type=float, default=1.0,
                        help="Multiplier applied to every font size")
    parser.add_argument("--title", default=DEFAULT_TITLE,
                        help="Figure title; the run's forecasting window is appended to it "
                             "automatically. Pass an empty string for no title")
    args = parser.parse_args()
    build_charts(args.state_csv, args.state_dir, args.output_dir,
                 args.formats, args.font_scale, args.title)


if __name__ == "__main__":
    main()
