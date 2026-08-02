"""
Minimal runnable demo for `mpc_controller.MPCController`.

Warms up the controller's history from a real preprocessed session
(`data_preprocessed/data1.csv`), then runs a few receding-horizon control
steps toward user-specified setpoints. Since there is no closed-loop plant
model matching these target names available in this repo yet, each step's
"measurement" is taken from the model's own first predicted step under the
chosen control (a self-consistent proxy for the real process) - this is only
meant to exercise the controller end-to-end, not to evaluate control quality.

Usage:
    .venv/bin/python -m mpc2.run_mpc_demo
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mpc2.mpc_controller import MPCController

# One model per target. Only these three have been trained under the "simple"
# run group so far (see darts_logs/); train "simple_gran1_blain_*" (enable it
# in config.py's SIMPLE_MODEL_CONFIG first) to add it here too.
MODEL_NAMES = {
    "return": "simple_return_LinearRegression_I2_O2",
    "first_chamber_filling": "simple_first_chamber_filling_LinearRegression_I2_O2",
    "second_chamber_filling": "simple_second_chamber_filling_LinearRegression_I2_O2",
}

SETPOINTS = {
    "return": 20.0,
    "first_chamber_filling": 70.0,
    "second_chamber_filling": 70.0,
}

N_STEPS = 10


def main() -> None:
    controller = MPCController(MODEL_NAMES, darts_logs_dir=PROJECT_ROOT / "darts_logs")

    df = pd.read_csv(PROJECT_ROOT / "data_preprocessed" / "data1.csv")
    warmup_rows = df[controller.history_cols].iloc[: controller.max_input_chunk_length].to_dict("records")
    controller.seed_history(warmup_rows)

    print(f"{'step':>4} | {'separator_speed':>16} | {'fresh_feed_setpoint':>20} | targets (predicted next step)")
    for step in range(N_STEPS):
        result = controller.choose_action(SETPOINTS)
        control = result["control"]

        next_measurement = dict(control)
        for target, preds in result["predictions"].items():
            next_measurement[target] = float(preds[0])
        controller.update_history(next_measurement)

        targets_str = ", ".join(f"{t}={next_measurement[t]:.2f}" for t in SETPOINTS)
        print(f"{step:>4} | {control['separator_speed']:>16.1f} | {control['fresh_feed_setpoint']:>20.1f} | {targets_str}")

    print("\nSetpoints were:", SETPOINTS)


if __name__ == "__main__":
    main()
