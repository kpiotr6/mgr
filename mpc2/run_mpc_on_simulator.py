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

Usage:
    .venv/bin/python -m mpc2.run_mpc_on_simulator
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# "mae": each target competes in its own units (raw-scale targets like
# gran1_blain then dominate the combined cost unless you also pass weights).
# "mape": scale-free (100 * |pred-setpoint|/|setpoint|), so all targets
# compete on comparable footing without hand-tuned weights.
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
    "separator_speed": 5.0,
    "fresh_feed_setpoint": 5.0,
}

NOMINAL_FEED_TH = 90.0   # t/h - used during physical warmup and history seeding
NOMINAL_RPM = 150.0      # matches CementMillSimulator's nominal_rpm

FINE_DT_MIN = 1.0        # minutes per simulator step - also the MPC decision period now
BLOCK_SIZE = int(round(STEP_MINUTES / FINE_DT_MIN))  # raw 1-min samples pooled into one model-step row

# Total simulated minutes to run under active MPC control (one decision per minute).
# NOTE: at ~1-2s per decision (batched grid search across all target models),
# 500 minutes takes on the order of 10-15 minutes of wall-clock compute.
N_CONTROLLED_MINUTES = 100

LOG_KEYS = ("return", "first_chamber_filling", "second_chamber_filling",
            "separator_speed", "fresh_feed_setpoint", "gran1_blain")


def _map_outputs_to_targets(out: dict, H_cap: float) -> dict:
    return {
        "return": float(out["Mr"]),
        "first_chamber_filling": float(np.clip(out["H1"] / H_cap * 100.0, 0.0, 100.0)),
        "second_chamber_filling": float(np.clip(out["H2"] / H_cap * 100.0, 0.0, 100.0)),
        "gran1_blain": float(out["blaine"]),
    }


def _advance_one_minute(sim, state, t_now, control):
    """Advance the simulator by exactly one minute under `control`, returning
    the new state/time and the raw (1-minute resolution) measurement row."""
    sim.set_rotor_speed(control["separator_speed"])
    Mc0 = control["fresh_feed_setpoint"] / 60.0  # t/h -> t/min

    state = sim.step(state, dt=FINE_DT_MIN, Mc=Mc0)
    t_now += FINE_DT_MIN
    out = sim.instantaneous_outputs(state)
    row = _map_outputs_to_targets(out, sim.H_cap)
    row["separator_speed"] = control["separator_speed"]
    row["fresh_feed_setpoint"] = control["fresh_feed_setpoint"]
    return state, t_now, row


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


def main():
    nominal_control = {"separator_speed": NOMINAL_RPM, "fresh_feed_setpoint": NOMINAL_FEED_TH}
    controller = MPCController(
        MODEL_NAMES,
        darts_logs_dir=PROJECT_ROOT / "darts_logs",
        weights=TARGET_WEIGHTS,
        error_metric=ERROR_METRIC,
        tie_break_tolerance=TIE_BREAK_TOLERANCE,
        move_penalty=MOVE_PENALTY_WEIGHTS,
        initial_control=nominal_control,
    )

    sim = CementMillSimulator()
    sim.set_rotor_speed(NOMINAL_RPM)

    print("Warming up the simulator to a steady state...")
    Mc_base = NOMINAL_FEED_TH / 60.0
    warmup = sim.simulate((0, 300.0), lambda t: Mc_base, t_eval=np.linspace(280.0, 300.0, 5))
    state = warmup.y[:, -1]

    t_now = 0.0
    log = {"t": [], **{k: [] for k in LOG_KEYS}}

    def _record(t, row):
        log["t"].append(t)
        for k in LOG_KEYS:
            log[k].append(row[k])

    raw_window = controller.max_input_chunk_length * BLOCK_SIZE
    raw_buffer: deque = deque(maxlen=raw_window)

    print(f"Seeding {raw_window} raw 1-minute step(s) at nominal control...")
    for _ in range(raw_window):
        state, t_now, row = _advance_one_minute(sim, state, t_now, nominal_control)
        raw_buffer.append(row)
        _record(t_now, row)

    seed_end_t = t_now
    print(f"Running {N_CONTROLLED_MINUTES} MPC-controlled 1-minute steps...")
    for minute in range(N_CONTROLLED_MINUTES):
        controller.set_history(_pooled_history(raw_buffer, controller.max_input_chunk_length))
        result = controller.choose_action(SETPOINTS)
        control = result["control"]

        state, t_now, row = _advance_one_minute(sim, state, t_now, control)
        raw_buffer.append(row)
        _record(t_now, row)

        if minute % 10 == 0 or minute == N_CONTROLLED_MINUTES - 1:
            error_str = ", ".join(f"{k}={v:.2f}" for k, v in result["error"].items())
            print(f"  minute {minute:>4}  t={t_now:6.0f} min  "
                  f"separator_speed={control['separator_speed']:6.1f}  "
                  f"fresh_feed_setpoint={control['fresh_feed_setpoint']:6.1f}  {ERROR_METRIC}: {error_str}")

    # ------------------------------------------------------------------
    fig, axes = plt.subplots(6, 1, figsize=(10, 16), sharex=True)
    fig.suptitle(
        "MPC controller closed-loop on CementMillSimulator\n"
        "(a new control is chosen every simulated minute; models still see "
        "5-minute mean-pooled history)",
        fontsize=12,
    )

    target_axes = [
        ("return", "Return flow $M_R$ (t/h)"),
        ("first_chamber_filling", "1st chamber filling (%)"),
        ("second_chamber_filling", "2nd chamber filling (%)"),
        ("gran1_blain", "Blaine gran1 (cm$^2$/g)"),
    ]
    for ax, (key, ylabel) in zip(axes[:4], target_axes):
        ax.plot(log["t"], log[key], color="tab:blue", lw=1.2, label=key)
        ax.axhline(SETPOINTS[key], color="tab:red", ls="--", lw=1.3, label="setpoint")
        ax.axvline(seed_end_t, color="gray", ls=":", lw=1, label="MPC control starts")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    ax = axes[4]
    ax.step(log["t"], log["separator_speed"], where="post", color="tab:purple", lw=1.0)
    ax.axvline(seed_end_t, color="gray", ls=":", lw=1)
    ax.set_ylabel("Separator speed (rpm)")
    ax.grid(alpha=0.3)

    ax = axes[5]
    ax.step(log["t"], log["fresh_feed_setpoint"], where="post", color="tab:orange", lw=1.0)
    ax.axvline(seed_end_t, color="gray", ls=":", lw=1)
    ax.set_ylabel("Fresh feed (t/h)")
    ax.set_xlabel("time (min)")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUTPUT_DIR / "fig_mpc_closed_loop.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
