"""Run a closed-loop MPC simulation for 60 steps and report tracking metrics.

Example:
  python -m mpc.run_closed_loop \
    --model-name NeuralForecast_Nhits_I60_O30 \
    --model-class NeuralForecastModel \
    --forward-horizon 30 \
    --grid-n 5

If `darts_logs/<model-name>/scalers.pkl` is missing, create it with:
  python -m mpc.fit_scalers \
    --model-name NeuralForecast_Nhits_I60_O30 \
    --input-cols clinker_1_feedrate separator_speed \
    --target-cols return first_chamber_filling second_chamber_filling gran1_blain
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mpc.simulator_headless import CementMillSim
from mpc.mpc_controller import (
    CONTROL_COLS,
    TARGET_COLS,
    DartsPredictor,
    DiscreteGridMPC,
    MPCConfig,
)


def mae(y: np.ndarray, sp: np.ndarray) -> float:
    return float(np.mean(np.abs(y - sp)))


def mse(y: np.ndarray, sp: np.ndarray) -> float:
    return float(np.mean((y - sp) ** 2))


def mape(y: np.ndarray, sp: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(sp), eps)
    return float(np.mean(np.abs((y - sp) / denom)) * 100.0)


def _parse_setpoints(s: str | None) -> dict[str, float] | None:
    if s is None:
        return None
    if Path(s).exists():
        return json.loads(Path(s).read_text())
    return json.loads(s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-class", default="NeuralForecastModel")
    parser.add_argument("--model-dir", default="darts_logs")
    parser.add_argument("--scalers-path", default=None)

    parser.add_argument("--forward-horizon", type=int, default=30, choices=[15, 30, 60])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=None, help="Defaults to model input_chunk_length")
    parser.add_argument("--grid-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--setpoints",
        default=None,
        help="JSON string or path to JSON with keys: return, first_chamber_filling, second_chamber_filling, gran1_blain",
    )

    parser.add_argument("--save-csv", default="outputs/mpc_closed_loop.csv")

    args = parser.parse_args()

    predictor = DartsPredictor(
        model_name=args.model_name,
        model_class_name=args.model_class,
        model_dir=args.model_dir,
        scalers_path=args.scalers_path,
        target_cols=TARGET_COLS,
        control_cols=CONTROL_COLS,
    )

    warmup = int(args.warmup) if args.warmup is not None else int(predictor.input_chunk_length)

    sim = CementMillSim(seed=args.seed)

    # Seed history with a constant operating point.
    u0 = {"clinker_1_feedrate": 60.0, "separator_speed": 540.0}

    y_hist: list[list[float]] = []
    u_hist: list[list[float]] = []

    for _ in range(warmup):
        y = sim.step(**u0)
        y_hist.append([y[c] for c in TARGET_COLS])
        u_hist.append([u0[c] for c in CONTROL_COLS])

    setpoints = _parse_setpoints(args.setpoints)
    if setpoints is None:
        # Default: regulate to the current operating point.
        last = y_hist[-1]
        setpoints = {c: float(v) for c, v in zip(TARGET_COLS, last)}

    mpc = DiscreteGridMPC(
        predictor=predictor,
        setpoints=setpoints,
        config=MPCConfig(forward_horizon=args.forward_horizon, grid_n=args.grid_n),
    )

    rows = []

    for k in range(args.steps):
        y_arr = np.asarray(y_hist, dtype=float)
        u_arr = np.asarray(u_hist, dtype=float)

        action = mpc.choose_action(y_hist=y_arr, u_hist=u_arr)
        u_apply = {
            "clinker_1_feedrate": float(action["clinker_1_feedrate"]),
            "separator_speed": float(action["separator_speed"]),
        }

        y = sim.step(**u_apply)

        y_hist.append([y[c] for c in TARGET_COLS])
        u_hist.append([u_apply[c] for c in CONTROL_COLS])

        row = {
            "step": k,
            "cost": float(action["cost"]),
            **{f"u_{c}": u_apply[c] for c in CONTROL_COLS},
            **{f"y_{c}": y[c] for c in TARGET_COLS},
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.save_csv, index=False)

    # Metrics: tracking error vs setpoints over the controlled steps.
    sp_vec = np.array([setpoints[c] for c in TARGET_COLS], dtype=float)
    y_mat = df[[f"y_{c}" for c in TARGET_COLS]].to_numpy(dtype=float)
    sp_mat = np.tile(sp_vec.reshape(1, -1), (len(y_mat), 1))

    metrics = {}
    for i, c in enumerate(TARGET_COLS):
        y_c = y_mat[:, i]
        sp_c = sp_mat[:, i]
        metrics[c] = {
            "MAE": mae(y_c, sp_c),
            "MSE": mse(y_c, sp_c),
            "MAPE": mape(y_c, sp_c),
        }

    metrics["overall"] = {
        "MAE": mae(y_mat, sp_mat),
        "MSE": mse(y_mat, sp_mat),
        "MAPE": mape(y_mat, sp_mat),
    }

    print("Setpoints:")
    print(json.dumps(setpoints, indent=2, sort_keys=True))
    print("\nTracking metrics (vs setpoints):")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nSaved trajectory CSV: {args.save_csv}")


if __name__ == "__main__":
    main()
