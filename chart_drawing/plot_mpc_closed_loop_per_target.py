"""Closed-loop MPC trajectories for the per-target setpoint sweep, one figure
per target.

Reads the ``mpc_state_per_minute_<target>.csv`` files written by
``mpc2/run_mpc_on_simulator.py --per-target``. That sweep does one closed-loop
run per entry in ``SETPOINTS``: in the run for target *T*, *T* keeps its
assigned setpoint while every other target is pinned to its own value at the
end of the warm-up/seeding phase. Every run therefore starts from an identical
plant state with exactly one non-zero error, which isolates that target's
closed-loop response -- whatever the controller does is attributable to *T*.

There are four targets, so this writes four figures:

    fig_mpc_closed_loop_target_return.png
    fig_mpc_closed_loop_target_first_chamber_filling.png
    fig_mpc_closed_loop_target_second_chamber_filling.png
    fig_mpc_closed_loop_target_gran1_blain.png

Panel layout, colours, setpoint labelling and seeding-band shading are shared
verbatim with ``plot_mpc_closed_loop.py`` -- the two families of figure sit next
to each other in the thesis and must be read the same way. The one thing this
script adds is telling the reader *which* of the four panels is the one being
driven: the target under test is titled in bold ink, the other three are muted
and marked as held, so a reader flicking between the four figures can see at a
glance what changed. Without that mark the four figures look nearly identical,
since three of the four setpoint lines simply sit on top of their trace.

Panels are titled with the *exact* variable names used in ``config.py`` and the
MPC runner (``first_chamber_filling``, never "1st chamber").

Usage:
    .venv/bin/python chart_drawing/plot_mpc_closed_loop_per_target.py
    .venv/bin/python chart_drawing/plot_mpc_closed_loop_per_target.py --state-dir mpc2
    .venv/bin/python chart_drawing/plot_mpc_closed_loop_per_target.py \
        --state-csv mpc2/mpc_state_per_minute_return.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_mpc_closed_loop import (  # noqa: E402
    FONT_SIZES, INK, INK_MUTED, STEM_PREFIX, TARGET_PANELS,
    build_figure, load_state, save,
)

DEFAULT_STATE_DIR = "./outputs_ready"
DEFAULT_OUTPUT_DIR = "./chart_drawing/mpc_closed_loop_per_target_charts"

# The four targets the sweep runs over, in the same reading order the panels
# use, so the figures come out in a predictable order.
TARGETS = [key for key, _unit in TARGET_PANELS]

DEFAULT_TITLE = "MPC controller closed-loop"

# Marks appended to a panel title. Kept short: the variable names are already
# long, and the title line is what sets the panel width.
DRIVEN_MARK = "  · target under test"
HELD_MARK = "  · held at warm-up value"

# A setpoint within this fraction of a target's own range counts as "sitting on
# the trace", i.e. held rather than driven. Used only as a cross-check on the
# file name (see driven_target); generous, because a held target still drifts a
# little once the controller starts moving the controls for the driven one.
HELD_TOLERANCE = 0.02


def driven_target(path: Path, frame: pd.DataFrame) -> str | None:
    """Which target this run was driving.

    Taken from the file name suffix (``..._return.csv`` -> ``return``), which is
    what the runner writes. The frame is used only to cross-check that choice:
    the driven target is the one whose setpoint is *not* where the plant already
    was when MPC took over. A mismatch means the file was renamed, so the name
    is not trusted and the frame decides.
    """
    from_name = None
    stem = path.stem
    if stem.startswith(f"{STEM_PREFIX}_"):
        candidate = stem[len(STEM_PREFIX) + 1:]
        if candidate in TARGETS:
            from_name = candidate

    from_frame = _driven_from_frame(frame)
    if from_frame is None:
        return from_name
    if from_name is not None and from_name != from_frame:
        print(f"  note: {path.name} is named for '{from_name}' but its setpoints "
              f"say '{from_frame}' is the driven target; using '{from_frame}'")
    return from_frame


def _driven_from_frame(frame: pd.DataFrame) -> str | None:
    """The target whose setpoint is furthest from where the plant started.

    Scored on each target's own scale (the four differ by two orders of
    magnitude), against the last open-loop seeding row - the plant state every
    run in the sweep starts from. Returns ``None`` when nothing clears
    ``HELD_TOLERANCE``, i.e. this does not look like a per-target run at all.
    """
    seeding = frame[frame["phase"] == "seeding"] if "phase" in frame.columns else frame
    if seeding.empty:
        seeding = frame

    best, best_score = None, 0.0
    for target in TARGETS:
        setpoint_col = f"{target}_setpoint"
        if target not in frame.columns or setpoint_col not in frame.columns:
            continue
        values = frame[target].to_numpy(dtype=float)
        span = float(values.max() - values.min())
        scale = span if span > 0 else max(abs(float(values[0])), 1.0)
        offset = abs(float(frame[setpoint_col].iloc[-1]) - float(seeding[target].iloc[-1]))
        score = offset / scale
        if score > best_score:
            best, best_score = target, score
    return best if best_score > HELD_TOLERANCE else None


def mark_roles(figure, driven: str | None, fonts: dict) -> None:
    """Retitle the target panels to say which one is driven and which are held.

    Done by rewriting the titles `plot_mpc_closed_loop.style_axis` already set,
    rather than by reimplementing the panel drawing: everything about how the
    panels look has to stay identical to the other closed-loop figures, and the
    only difference here is this annotation.
    """
    if driven is None:
        return
    for axis in figure.axes:
        key = axis.get_title(loc="left")
        if key not in TARGETS:
            continue      # control panels and hidden axes keep their titles
        is_driven = key == driven
        axis.set_title(key + (DRIVEN_MARK if is_driven else HELD_MARK),
                       loc="left", fontsize=fonts["panel"],
                       color=INK if is_driven else INK_MUTED,
                       fontweight="bold" if is_driven else "normal", pad=10)


def figure_title(base: str, driven: str | None) -> str | None:
    """Base title, extended with the target this run was driving."""
    if not base:
        return None
    if driven is None:
        return base
    return f"{base} — driving {driven}"


def resolve_inputs(state_csvs, state_dir) -> list[Path]:
    """Explicit ``--state-csv`` files if given, else the sweep's four runs.

    Ordered by ``TARGETS`` rather than by name, so the figures come out in the
    same order as the panels instead of alphabetically.
    """
    if state_csvs:
        return [Path(p) for p in state_csvs]
    directory = Path(state_dir)
    paths = [directory / f"{STEM_PREFIX}_{target}.csv" for target in TARGETS]
    found = [p for p in paths if p.exists()]
    if not found:
        raise SystemExit(
            f"No {STEM_PREFIX}_<target>.csv files found in {state_dir}. "
            "Generate them with: .venv/bin/python -m mpc2.run_mpc_on_simulator --per-target"
        )
    for path in paths:
        if path not in found:
            print(f"  note: {path.name} not found, skipping")
    return found


def build_charts(state_csvs, state_dir, output_dir, formats, font_scale, title) -> None:
    fonts = {role: size * font_scale for role, size in FONT_SIZES.items()}
    for path in resolve_inputs(state_csvs, state_dir):
        frame = load_state(path)
        driven = driven_target(path, frame)
        figure = build_figure(frame, fonts, figure_title(title, driven))
        mark_roles(figure, driven, fonts)
        name = driven or path.stem
        paths = [Path(output_dir) / f"fig_mpc_closed_loop_target_{name}.{ext}"
                 for ext in formats]
        save(figure, paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the per-target MPC setpoint sweep: one closed-loop figure per "
                    "target, with the target under test marked."
    )
    parser.add_argument("--state-csv", nargs="+", default=None,
                        help=f"One or more {STEM_PREFIX}_<target>.csv files (one figure "
                             "each); defaults to all four runs in --state-dir")
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR,
                        help=f"Directory holding {STEM_PREFIX}_<target>.csv (used when "
                             "--state-csv is not given)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save charts")
    parser.add_argument("--formats", nargs="+", default=["png"],
                        help="Output formats to write for each chart (e.g. png pdf svg)")
    parser.add_argument("--font-scale", type=float, default=1.0,
                        help="Multiplier applied to every font size")
    parser.add_argument("--title", default=DEFAULT_TITLE,
                        help="Figure title; the driven target is appended to it "
                             "automatically. Pass an empty string for no title")
    args = parser.parse_args()
    build_charts(args.state_csv, args.state_dir, args.output_dir,
                 args.formats, args.font_scale, args.title)


if __name__ == "__main__":
    main()
