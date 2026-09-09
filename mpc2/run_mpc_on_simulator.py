"""
Run `MPCController` in closed loop against the physics-based
`CementMillSimulator` and plot the resulting trajectories, with setpoints
overlaid.

Mapping between the simulator's physical outputs and the darts models'
target names (there is no literal 1:1 correspondence in the training data,
so this is a reasonable-but-arbitrary bridge purely for demonstration):
  - "return"                  <- M_R, recirculated (reject) flow (t/h)
  - "first_chamber_filling"   <- compartment-1 hold-up H1, as % of capacity
  - "second_chamber_filling"  <- compartment-2 hold-up H2, as % of capacity

Controls:
  - "separator_speed"    -> simulator rotor speed (rpm), via set_rotor_speed()
  - "fresh_feed_setpoint" -> simulator fresh feed rate M_C (t/h)

The darts models operate on 5-minute steps, but the simulator - and the MPC
decision loop below - run at 1-minute resolution: a new control is chosen
*every simulated minute*. To feed the 5-minute-resolution models without
waiting 5 minutes between decisions, the controller's history is rebuilt
every minute from a *rolling* window of the trailing raw 1-minute
simulator readings, mean-pooled into 5-minute-equivalent blocks (see
`_pooled_history`). This is still "mean subsampling" as before, just
recomputed from scratch each minute instead of accumulated once every 5
minutes, so the model always sees its usual step size while the controller
gets to replan every minute.

Because replanning every minute would otherwise let the controller flip
between near-equally-good grid points every step (chattering), that
"1-minute" is where `MPCController`'s `tie_break_tolerance` mechanism does
its job: it prefers whichever near-optimal candidate is closest to the
control already applied.

Outputs (all written next to this script):
  - mpc_state_per_minute.csv - the state at *every* simulated minute: targets,
    setpoints and errors, applied controls, the simulator's raw physical
    outputs, and the MPC decision cost/predicted errors for that minute.
  - mpc_closed_loop_iae.csv  - per-target closed-loop IAE summary.
  - fig_mpc_closed_loop.png  - the 6-panel trajectory figure.

Usage:
    .venv/bin/python -m mpc2.run_mpc_on_simulator
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure typography: the closed-loop plot is meant to be readable when it is
# scaled down to fit a page/slide, so everything is well above matplotlib's
# 10pt default. Axis labels carry the exact variable names, which are long,
# hence the separate (smaller) legend size.
LABEL_FONTSIZE = 17
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 13
plt.rcParams.update({
    "font.size": TICK_FONTSIZE,
    "axes.labelsize": LABEL_FONTSIZE,
    "axes.titlesize": LABEL_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
})

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mpc2.cement_mill_sim import CementMillSimulator
from mpc2.mpc_controller import MPCController, STEP_MINUTES

OUTPUT_DIR = SCRIPT_DIR

MODEL_NAMES = {
    "return": "simple_return_LinearRegression_I6_O3",
    "first_chamber_filling": "simple_first_chamber_filling_NeuralForecast_TSMixer_I2_O3",
    "second_chamber_filling": "simple_second_chamber_filling_NeuralForecast_TSMixer_I2_O3",
    "gran1_blain": "simple_gran1_blain_LinearRegression_I6_O3",
}

SETPOINTS = {
    "return": 80.0,
    "first_chamber_filling": 50.0,
    "second_chamber_filling": 50.0,
    "gran1_blain": 5500.0,
}

IS_NAIVE = False

# Per-target setpoint sweep. When True (or with --per-target on the command
# line), the script does not do one run against the full SETPOINTS vector.
# Instead it does *one run per entry in SETPOINTS*: in the run for target T,
# T keeps its assigned setpoint from SETPOINTS while every other target gets a
# setpoint equal to its own value at the end of the warm-up/seeding phase -
# i.e. "hold everything else where the plant already is, and only ask the
# controller to move T".
#
# This isolates each target's closed-loop response: any control action the MPC
# takes is attributable to T alone, because the other targets start with zero
# error and are only penalised for drifting away. It also makes the per-target
# IAE numbers comparable, since every run begins from the identical plant
# state (the warm-up and seeding phases are executed once and shared).
#
# Each run writes its own suffixed outputs, e.g. for target "return":
#   mpc_state_per_minute_return.csv / mpc_closed_loop_iae_return.csv /
#   fig_mpc_closed_loop_return.png
PER_TARGET_RUNS = True

# Toggle for adding small Gaussian noise to target variables after warmup
ADD_NOISE = False
NOISE_STD = {
    "return": 1.0,
    "first_chamber_filling": 0.5,
    "second_chamber_filling": 0.5,
    "gran1_blain": 50.0,
}

# "mae": each target competes in its own units (raw-scale targets like
# gran1_blain then dominate the combined cost unless you also pass weights).
# "mape": scale-free (100 * |pred-setpoint|/|setpoint|), so all targets
# compete on comparable footing without hand-tuned weights.
# "itae": like "mae", but each horizon step's error is weighted by its
# elapsed time, so candidates whose error persists to the end of the horizon
# are punished harder than ones that only overshoot transiently. Same scale
# as "mae", so TARGET_WEIGHTS below carry over unchanged.
ERROR_METRIC = "mae"

# Per-target weights applied to each target's error before summing into the
# candidate's total cost (see MPCController.choose_action). Raise a target's
# weight to make the controller prioritize it more heavily over the others;
# defaults to 1.0 for any target not listed here. With ERROR_METRIC="mae",
# gran1_blain's error is naturally ~50-100x larger in raw units than the
# filling percentages, so its weight is scaled down to put all four targets
# on a comparable footing (tune to taste).
TARGET_WEIGHTS = {
    "return": 1.0,
    "first_chamber_filling": 1.0,
    "second_chamber_filling": 1.0,
    "gran1_blain": 0.02,
}

CONTROL_STEPS = 10.0

# Candidates within this fraction of the best cost are treated as ties and
# broken by proximity to the currently-applied control - see module docstring.
TIE_BREAK_TOLERANCE = 0.01

# Per-control-variable weight punishing candidates for deviating from the
# currently-applied control (normalized by that control's bounds range and
# added directly to the cost - see MPCController._move_penalty). Unlike
# TIE_BREAK_TOLERANCE, this actually competes with the target error terms:
# raise it to make the controller more reluctant to swing the controls, even
# when a bigger swing would predict a somewhat lower error.
MOVE_PENALTY_WEIGHTS = {
    "separator_speed": 10.0,
    "fresh_feed_setpoint": 10.0,
}


NOMINAL_FEED_TH = 130.0  # t/h - used during physical warmup and history seeding
                         # (also the upper `fresh_feed_setpoint` control bound)
NOMINAL_RPM = 205.0      # rpm - CementMillSimulator's own nominal_rpm is 150,
                         # which stays the sep_bias=1 reference; this is the
                         # speed we actually hold during warm-up/seeding.

WARMUP_MINUTES = 900.0

FINE_DT_MIN = 1.0        # minutes per simulator step - also the MPC decision period now
BLOCK_SIZE = int(round(STEP_MINUTES / FINE_DT_MIN))  # raw 1-min samples pooled into one model-step row

# Total simulated minutes to run under active MPC control (one decision per minute).
# NOTE: at ~1-2s per decision (batched grid search across all target models),
# 500 minutes takes on the order of 10-15 minutes of wall-clock compute.
N_CONTROLLED_MINUTES = 200

LOG_KEYS = ("return", "first_chamber_filling", "second_chamber_filling",
            "separator_speed", "fresh_feed_setpoint", "gran1_blain")


def _map_outputs_to_targets(out: dict, H_cap: float) -> dict:
    return {
        "return": float(out["Mr"]),
        "first_chamber_filling": float(np.clip(out["H1"] / H_cap * 100.0, 0.0, 100.0)),
        "second_chamber_filling": float(np.clip(out["H2"] / H_cap * 100.0, 0.0, 100.0)),
        "gran1_blain": float(out["blaine"]),
    }


def _advance_one_minute(sim, state, t_now, control, apply_noise=False):
    """Advance the simulator by exactly one minute under `control`, returning
    the new state/time, the raw (1-minute resolution) measurement row, and the
    simulator's own instantaneous outputs (kept separate from `row` because
    `row` is what gets mean-pooled into the controller's history, and must
    hold exactly the model/control columns and nothing else)."""
    sim.set_rotor_speed(control["separator_speed"])
    Mc0 = control["fresh_feed_setpoint"] / 60.0  # t/h -> t/min

    state = sim.step(state, dt=FINE_DT_MIN, Mc=Mc0)
    t_now += FINE_DT_MIN
    out = sim.instantaneous_outputs(state)
    row = _map_outputs_to_targets(out, sim.H_cap)

    if ADD_NOISE and apply_noise:
        for k, std in NOISE_STD.items():
            if k in row:
                row[k] += np.random.normal(0.0, std)

    row["separator_speed"] = control["separator_speed"]
    row["fresh_feed_setpoint"] = control["fresh_feed_setpoint"]
    return state, t_now, row, out


def _pooled_history(raw_buffer: deque, n_blocks: int) -> list[dict]:
    """Mean-pool the trailing `n_blocks * BLOCK_SIZE` raw 1-minute rows into
    `n_blocks` consecutive 5-minute-equivalent rows (oldest first) - the
    "one row per model step" history `MPCController` expects."""
    rows = list(raw_buffer)[-n_blocks * BLOCK_SIZE:]
    pooled = []
    for i in range(n_blocks):
        chunk = rows[i * BLOCK_SIZE: (i + 1) * BLOCK_SIZE]
        pooled.append({k: float(np.mean([r[k] for r in chunk])) for k in chunk[0]})
    return pooled


def _plot_closed_loop(log: dict, seed_end_t: float,
                      setpoints: Mapping[str, float], suffix: str = "") -> None:
    """Render the 6-panel closed-loop figure.

    `setpoints` is passed in rather than read from the module-level SETPOINTS
    because the per-target sweep (see PER_TARGET_RUNS) gives each run its own
    setpoint vector. `suffix` is appended to the output filename so the sweep's
    runs do not overwrite each other.

    Each panel is identified by the *exact* variable name used everywhere else
    (MODEL_NAMES / SETPOINTS / the controller's history columns), so the figure
    and the code/tables can be cross-read without a translation step. Those
    names are too long to sit rotated in the y-label at this font size - they
    overflow the panel height and collide between panels - so the name goes in
    a left-aligned panel title and the y-label carries only the unit.
    """
    fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(
        "MPC controller closed-loop on CementMillSimulator\n"
        "(a new control is chosen every simulated minute; models still see "
        "5-minute mean-pooled history)",
        fontsize=LABEL_FONTSIZE,
    )

    target_axes = [
        ("return", "t/h"),
        ("first_chamber_filling", "%"),
        ("second_chamber_filling", "%"),
        ("gran1_blain", "cm$^2$/g"),
    ]
    for i, (ax, (key, unit)) in enumerate(zip(axes[:4], target_axes)):
        ax.plot(log["t"], log[key], color="tab:blue", lw=1.8)
        ax.axhline(setpoints[key], color="tab:red", ls="--", lw=1.8, label="setpoint")
        ax.axvline(seed_end_t, color="gray", ls=":", lw=1.5, label="MPC control starts")
        ax.set_title(key, loc="left", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3)
        # Styling is identical across the target panels, so one legend is
        # enough - four of them just cover data.
        if i == 0:
            ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE)

    control_axes = [
        ("separator_speed", "rpm", "tab:purple"),
        ("fresh_feed_setpoint", "t/h", "tab:orange"),
    ]
    for ax, (key, unit, color) in zip(axes[4:], control_axes):
        ax.step(log["t"], log[key], where="post", color=color, lw=1.6)
        ax.axvline(seed_end_t, color="gray", ls=":", lw=1.5)
        ax.set_title(key, loc="left", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (min)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUTPUT_DIR / f"fig_mpc_closed_loop{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _closed_loop_iae(log: dict, seed_end_t: float,
                     setpoints: Mapping[str, float]) -> list[dict]:
    """Closed-loop IAE (integral of absolute error) per target over the whole
    logged run.

    This is the *outer-loop* performance measure, and is deliberately not the
    same thing as `MPCController.error_metric`: the controller's metric scores
    a candidate's predicted N-step horizon at each decision, whereas this
    integrates the error the plant actually realized against the flat
    setpoint, across the entire simulated timeline:

        IAE_target = integral |y(t) - setpoint| dt   [target units x minutes]

    Integrated with the trapezoid rule over `log["t"]` (minutes). Reported for
    two windows: "full" (everything logged, including the nominal-control
    seeding phase, which is what "whole time" means here) and "controlled"
    (only t >= seed_end_t, i.e. after MPC took over) - the seeding phase runs
    open-loop at nominal control, so it charges the controller for error it
    was never given the chance to correct. `iae_per_minute` divides by the
    window length, making the two windows (and runs of different length)
    directly comparable; it is exactly the time-averaged absolute error.
    """
    t = np.asarray(log["t"], dtype=float)
    rows = []

    def _integrate(err: np.ndarray, times: np.ndarray) -> tuple[float, float]:
        if len(times) < 2:
            return 0.0, 0.0
        iae = float(np.trapezoid(err, times))
        return iae, iae / (times[-1] - times[0])

    controlled = t >= seed_end_t
    weighted = {"full": 0.0, "controlled": 0.0}
    for target, setpoint in setpoints.items():
        abs_err = np.abs(np.asarray(log[target], dtype=float) - setpoint)
        entry = {"target": target, "setpoint": setpoint}
        for window, mask in (("full", np.ones_like(t, dtype=bool)), ("controlled", controlled)):
            iae, per_min = _integrate(abs_err[mask], t[mask])
            entry[f"iae_{window}"] = iae
            entry[f"iae_per_minute_{window}"] = per_min
            weighted[window] += TARGET_WEIGHTS.get(target, 1.0) * iae
        rows.append(entry)

    # Same weighting the controller uses to combine targets into one cost, so
    # this single number is comparable across runs that tune the controller.
    rows.append({
        "target": "TOTAL_weighted",
        "setpoint": float("nan"),
        "iae_full": weighted["full"],
        "iae_per_minute_full": weighted["full"] / (t[-1] - t[0]) if len(t) > 1 else 0.0,
        "iae_controlled": weighted["controlled"],
        "iae_per_minute_controlled": (
            weighted["controlled"] / (t[controlled][-1] - t[controlled][0])
            if controlled.sum() > 1 else 0.0
        ),
    })
    return rows


def _run_controlled(controller, sim, state, t_now, seed_history,
                    seed_log_rows, seed_end_t, setpoints, nominal_control,
                    suffix="", label=None):
    """Run the `N_CONTROLLED_MINUTES` closed-loop phase against `setpoints`,
    starting from an already warmed-up and seeded plant state, and write this
    run's three output files (suffixed with `suffix`).

    `state` / `seed_history` / `seed_log_rows` are the shared snapshot taken at
    the end of seeding; the caller is responsible for handing each run its own
    copies, since a run mutates all three. The controller is reused across runs
    (loading the darts models is by far the slowest part of startup), so its
    only cross-run state - `last_control` - is reset here.
    """
    if label:
        print(f"\n=== run: {label} ===")
        print("  setpoints: " + ", ".join(f"{k}={v:.2f}" for k, v in setpoints.items()))

    controller.last_control = dict(nominal_control)
    raw_buffer: deque = deque(seed_history, maxlen=controller.max_input_chunk_length * BLOCK_SIZE)

    log = {"t": [], **{k: [] for k in LOG_KEYS}}
    # One row per simulated minute, written out as a CSV at the end: the full
    # per-minute state of the run (targets, controls, the simulator's own
    # physical outputs, and - once MPC takes over - the decision that produced
    # that minute's control). `log` above stays the plotting-only subset.
    state_rows: list[dict] = []

    def _record(t, row, out, phase, result=None):
        log["t"].append(t)
        for k in LOG_KEYS:
            log[k].append(row[k])

        rec = {"minute": int(round(t / FINE_DT_MIN)), "t_min": float(t), "phase": phase}
        rec.update({k: float(row[k]) for k in LOG_KEYS})
        for target, setpoint in setpoints.items():
            rec[f"{target}_setpoint"] = setpoint
            rec[f"{target}_error"] = float(row[target]) - setpoint
        # Raw simulator outputs (Mm, Mp, Mr, blaine, H1, H2, H, ...) under a
        # "sim_" prefix so they cannot collide with the target names above,
        # which are a mapped/rescaled view of the same quantities.
        rec.update({f"sim_{k}": float(v) for k, v in out.items()})
        rec["mpc_cost"] = float(result["cost"]) if result is not None else float("nan")
        for target in setpoints:
            rec[f"mpc_predicted_error_{target}"] = (
                float(result["error"][target])
                if result is not None and target in result["error"]
                else float("nan")
            )
        state_rows.append(rec)

    # Replay the shared seeding phase into this run's log, so the "full" IAE
    # window and the figure still cover it - it is scored against *this* run's
    # setpoints even though the plant trajectory through it is identical.
    for seed_t, seed_row, seed_out in seed_log_rows:
        _record(seed_t, seed_row, seed_out, phase="seeding")

    print(f"  Running {N_CONTROLLED_MINUTES} MPC-controlled 1-minute steps...")
    for minute in range(N_CONTROLLED_MINUTES):
        controller.set_history(_pooled_history(raw_buffer, controller.max_input_chunk_length))
        result = controller.choose_action(setpoints)
        control = result["control"]

        # apply_noise=True during active control phase
        state, t_now, row, out = _advance_one_minute(sim, state, t_now, control, apply_noise=True)
        raw_buffer.append(row)
        _record(t_now, row, out, phase="controlled", result=result)

        if minute % 10 == 0 or minute == N_CONTROLLED_MINUTES - 1:
            error_str = ", ".join(f"{k}={v:.2f}" for k, v in result["error"].items())
            print(f"    minute {minute:>4}  t={t_now:6.0f} min  "
                  f"separator_speed={control['separator_speed']:6.1f}  "
                  f"fresh_feed_setpoint={control['fresh_feed_setpoint']:6.1f}  {ERROR_METRIC}: {error_str}")

    # ------------------------------------------------------------------
    state_path = OUTPUT_DIR / f"mpc_state_per_minute{suffix}.csv"
    pd.DataFrame(state_rows).to_csv(state_path, index=False)
    print(f"\n  Saved {state_path} ({len(state_rows)} minute(s) of state)")

    iae_rows = _closed_loop_iae(log, seed_end_t, setpoints)
    iae_path = OUTPUT_DIR / f"mpc_closed_loop_iae{suffix}.csv"
    pd.DataFrame(iae_rows).to_csv(iae_path, index=False)
    print(f"  Closed-loop IAE (error_metric={ERROR_METRIC}, {N_CONTROLLED_MINUTES} controlled minutes):")
    print(f"    {'target':<24} {'IAE full':>14} {'IAE controlled':>16} {'per-minute (ctrl)':>19}")
    for r in iae_rows:
        print(f"    {r['target']:<24} {r['iae_full']:>14.2f} {r['iae_controlled']:>16.2f} "
              f"{r['iae_per_minute_controlled']:>19.3f}")
    print(f"  Saved {iae_path}")

    _plot_closed_loop(log, seed_end_t, setpoints, suffix)
    return iae_rows


def _build_setpoint_plans(per_target: bool, held: Mapping[str, float]):
    """Return the list of (label, setpoints, suffix) runs to execute.

    With `per_target` off this is a single run against SETPOINTS as written.
    With it on there is one run per SETPOINTS entry: that entry keeps its
    assigned setpoint, every other target is pinned to `held` - its own value
    at the end of the warm-up/seeding phase - so only the one target starts
    with a non-zero error. See the PER_TARGET_RUNS comment above.
    """
    if not per_target:
        return [(None, dict(SETPOINTS), "")]

    plans = []
    for target in SETPOINTS:
        setpoints = {
            t: (float(SETPOINTS[t]) if t == target else float(held[t]))
            for t in SETPOINTS
        }
        plans.append((f"{target} @ {SETPOINTS[target]:g} (others held at warm-up value)",
                      setpoints, f"_{target}"))
    return plans


def main(per_target: bool = PER_TARGET_RUNS):
    nominal_control = {"separator_speed": NOMINAL_RPM, "fresh_feed_setpoint": NOMINAL_FEED_TH}
    controller = MPCController(
        MODEL_NAMES,
        darts_logs_dir=PROJECT_ROOT / "darts_logs",
        control_step=CONTROL_STEPS,
        weights=TARGET_WEIGHTS,
        error_metric=ERROR_METRIC,
        tie_break_tolerance=TIE_BREAK_TOLERANCE,
        move_penalty=MOVE_PENALTY_WEIGHTS,
        initial_control=nominal_control,
        naive=IS_NAIVE
    )

    sim = CementMillSimulator()
    sim.set_rotor_speed(NOMINAL_RPM)

    print("Warming up the simulator to a steady state...")
    Mc_base = NOMINAL_FEED_TH / 60.0
    warmup = sim.simulate((0, WARMUP_MINUTES), lambda t: Mc_base,
                          t_eval=np.linspace(WARMUP_MINUTES - 20.0, WARMUP_MINUTES, 5))
    state = warmup.y[:, -1]
    t_now = 0.0

    # Warm-up and seeding are setpoint-independent (they run open-loop at
    # `nominal_control`), so they are executed once and every run below starts
    # from this identical snapshot. That is what makes the per-target runs
    # comparable to each other.
    raw_window = controller.max_input_chunk_length * BLOCK_SIZE
    seed_history: list[dict] = []
    seed_log_rows: list[tuple] = []

    print(f"Seeding {raw_window} raw 1-minute step(s) at nominal control...")
    for _ in range(raw_window):
        # apply_noise=False during seeding phase
        state, t_now, row, out = _advance_one_minute(sim, state, t_now, nominal_control, apply_noise=False)
        seed_history.append(row)
        seed_log_rows.append((t_now, row, out))
    seed_end_t = t_now

    # The plant's own value for each target at the moment MPC takes over. In a
    # per-target run this is the setpoint handed to every target except the one
    # under test, so those start at exactly zero error.
    held = {t: float(seed_history[-1][t]) for t in SETPOINTS}
    print("Values at end of warm-up/seeding: "
          + ", ".join(f"{k}={v:.2f}" for k, v in held.items()))

    plans = _build_setpoint_plans(per_target, held)
    if per_target:
        print(f"\nPER-TARGET SWEEP: {len(plans)} run(s), one per SETPOINTS entry.")

    for label, setpoints, suffix in plans:
        _run_controlled(
            controller, sim, state.copy(), t_now,
            seed_history, seed_log_rows, seed_end_t,
            setpoints, nominal_control, suffix=suffix, label=label,
        )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--per-target", dest="per_target", action="store_true", default=None,
        help="Run once per SETPOINTS entry: that target keeps its assigned setpoint "
             "while every other target is pinned to its own value at the end of "
             "warm-up/seeding. Outputs are suffixed with the target name.",
    )
    group.add_argument(
        "--single", dest="per_target", action="store_false",
        help="Do one run against the full SETPOINTS vector (the default).",
    )
    args = parser.parse_args(argv)
    if args.per_target is None:
        args.per_target = PER_TARGET_RUNS
    return args


if __name__ == "__main__":
    main(per_target=_parse_args().per_target)
