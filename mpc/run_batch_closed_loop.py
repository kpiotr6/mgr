"""Batch runner for closed-loop MPC experiments.

Runs:
- All Darts models found under `darts_logs/<model_name>/` (requires `scalers.pkl`).
- A naive baseline predictor across all forward horizons.

Outputs:
- Per-run trajectory CSVs and metrics JSONs under `outputs/mpc_batch_closed_loop/`.
- A combined summary CSV at `outputs/mpc_batch_closed_loop/summary.csv`.

Example:
  python -m mpc.run_batch_closed_loop

  python -m mpc.run_batch_closed_loop \
    --model-dir darts_logs \
    --horizons 15,30,60 \
    --steps 60 \
    --grid-n 5 \
    --limit 3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


_HORIZON_RE = re.compile(r"_I(?P<i>\d+)_O(?P<o>\d+)(?:_|$)")


def _parse_horizons(s: str) -> list[int]:
    items = [p.strip() for p in s.split(",") if p.strip()]
    horizons: list[int] = []
    for it in items:
        try:
            horizons.append(int(it))
        except ValueError as e:
            raise ValueError(f"Invalid horizon '{it}'") from e
    if not horizons:
        raise ValueError("No horizons provided")
    for h in horizons:
        if h not in (15, 30, 60):
            raise ValueError("Horizons must be 15, 30, 60")
    return horizons


def _infer_warmup_from_model_name(model_name: str, default: int = 60) -> int:
    m = _HORIZON_RE.search(model_name)
    if not m:
        return int(default)
    return int(m.group("i"))


def _infer_forward_from_model_name(model_name: str) -> int | None:
    m = _HORIZON_RE.search(model_name)
    if not m:
        return None
    return int(m.group("o"))


def _safe_slug(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _run_one(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return int(p.returncode), str(p.stdout)


def _load_metrics_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="darts_logs")
    parser.add_argument(
        "--horizons",
        default="15,30,60",
        help="Comma-separated list of forward horizons (15,30,60).",
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--grid-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--setpoints",
        default=None,
        help="JSON string or path to JSON with setpoints (passed through to run_closed_loop).",
    )

    parser.add_argument(
        "--model-class",
        default="NeuralForecastModel",
        help=(
            "Default Darts model class for non-linear-regression models. "
            "(run_closed_loop will infer LinearRegressionModel for LinearRegression_* names)"
        ),
    )

    parser.add_argument(
        "--out-dir",
        default="outputs/mpc_batch_closed_loop",
        help="Output directory for per-run artifacts and summary.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of Darts models to run.",
    )
    parser.add_argument(
        "--naive-warmup",
        type=int,
        default=60,
        help="Warmup steps to pass explicitly for naive runs.",
    )
    parser.add_argument(
        "--warmup-default",
        type=int,
        default=60,
        help="Fallback warmup for Darts models when input length can't be inferred from the model name.",
    )

    args = parser.parse_args()

    horizons = _parse_horizons(args.horizons)

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Collect models: require directory + scalers.pkl to avoid noisy failures.
    model_names: list[str] = []
    if model_dir.exists():
        for p in sorted(model_dir.iterdir()):
            if not p.is_dir():
                continue
            if (p / "scalers.pkl").exists():
                model_names.append(p.name)

    if args.limit is not None:
        model_names = model_names[: int(args.limit)]

    summary_rows: list[dict[str, Any]] = []

    def add_row_common(row: dict[str, Any], *, rc: int, log: str) -> None:
        row["status"] = "ok" if rc == 0 else "error"
        row["returncode"] = int(rc)
        if rc != 0:
            # Keep the summary csv readable.
            row["error"] = log[-2000:]

    # 1) Darts runs.
    for model_name in model_names:
        inferred_forward = _infer_forward_from_model_name(model_name)
        if inferred_forward is not None:
            # Run only on the horizon the model was trained for (O{h}).
            if inferred_forward not in horizons:
                continue
            forward_horizons = [int(inferred_forward)]
        else:
            # Legacy / unknown naming: run across requested horizons.
            forward_horizons = horizons

        for h in forward_horizons:
            warmup = _infer_warmup_from_model_name(model_name, default=int(args.warmup_default))

            run_id = _safe_slug(f"darts__{model_name}__H{h}__W{warmup}__S{args.steps}__G{args.grid_n}__seed{args.seed}")
            traj_csv = runs_dir / f"{run_id}.csv"
            metrics_json = runs_dir / f"{run_id}.metrics.json"

            cmd = [
                sys.executable,
                "-m",
                "mpc.run_closed_loop",
                "--predictor",
                "darts",
                "--model-name",
                model_name,
                "--model-class",
                args.model_class,
                "--model-dir",
                str(model_dir),
                "--forward-horizon",
                str(h),
                "--steps",
                str(args.steps),
                "--grid-n",
                str(args.grid_n),
                "--seed",
                str(args.seed),
                "--warmup",
                str(warmup),
                "--save-csv",
                str(traj_csv),
                "--save-metrics-json",
                str(metrics_json),
            ]
            if args.setpoints is not None:
                cmd += ["--setpoints", str(args.setpoints)]

            rc, log = _run_one(cmd)

            row: dict[str, Any] = {
                "predictor": "darts",
                "model_name": model_name,
                "forward_horizon": int(h),
                "warmup": int(warmup),
                "steps": int(args.steps),
                "grid_n": int(args.grid_n),
                "seed": int(args.seed),
                "trajectory_csv": str(traj_csv),
                "metrics_json": str(metrics_json),
            }

            add_row_common(row, rc=rc, log=log)

            if rc == 0 and metrics_json.exists():
                payload = _load_metrics_json(metrics_json)
                overall = (payload.get("metrics") or {}).get("overall") or {}
                row["overall_MAE"] = overall.get("MAE")
                row["overall_MSE"] = overall.get("MSE")
                row["overall_MAPE"] = overall.get("MAPE")

                for c, m in (payload.get("metrics") or {}).items():
                    if c == "overall" or not isinstance(m, dict):
                        continue
                    row[f"{c}_MAE"] = m.get("MAE")
                    row[f"{c}_MSE"] = m.get("MSE")
                    row[f"{c}_MAPE"] = m.get("MAPE")

            summary_rows.append(row)

    # 2) Naive runs.
    for h in horizons:
        warmup = int(args.naive_warmup)

        run_id = _safe_slug(f"naive__H{h}__W{warmup}__S{args.steps}__G{args.grid_n}__seed{args.seed}")
        traj_csv = runs_dir / f"{run_id}.csv"
        metrics_json = runs_dir / f"{run_id}.metrics.json"

        cmd = [
            sys.executable,
            "-m",
            "mpc.run_closed_loop",
            "--predictor",
            "naive",
            "--forward-horizon",
            str(h),
            "--steps",
            str(args.steps),
            "--grid-n",
            str(args.grid_n),
            "--seed",
            str(args.seed),
            "--warmup",
            str(warmup),
            "--save-csv",
            str(traj_csv),
            "--save-metrics-json",
            str(metrics_json),
        ]
        if args.setpoints is not None:
            cmd += ["--setpoints", str(args.setpoints)]

        rc, log = _run_one(cmd)

        row = {
            "predictor": "naive",
            "model_name": "",
            "forward_horizon": int(h),
            "warmup": int(warmup),
            "steps": int(args.steps),
            "grid_n": int(args.grid_n),
            "seed": int(args.seed),
            "trajectory_csv": str(traj_csv),
            "metrics_json": str(metrics_json),
        }

        add_row_common(row, rc=rc, log=log)

        if rc == 0 and metrics_json.exists():
            payload = _load_metrics_json(metrics_json)
            overall = (payload.get("metrics") or {}).get("overall") or {}
            row["overall_MAE"] = overall.get("MAE")
            row["overall_MSE"] = overall.get("MSE")
            row["overall_MAPE"] = overall.get("MAPE")

            for c, m in (payload.get("metrics") or {}).items():
                if c == "overall" or not isinstance(m, dict):
                    continue
                row[f"{c}_MAE"] = m.get("MAE")
                row[f"{c}_MSE"] = m.get("MSE")
                row[f"{c}_MAPE"] = m.get("MAPE")

        summary_rows.append(row)

    # Write summary.
    summary_path = out_dir / "summary.csv"
    # Stable column order: common first, then metric columns.
    base_cols = [
        "predictor",
        "model_name",
        "forward_horizon",
        "warmup",
        "steps",
        "grid_n",
        "seed",
        "status",
        "returncode",
        "trajectory_csv",
        "metrics_json",
        "overall_MAE",
        "overall_MSE",
        "overall_MAPE",
        "error",
    ]

    extra_cols: list[str] = []
    for r in summary_rows:
        for k in r.keys():
            if k in base_cols or k in extra_cols:
                continue
            extra_cols.append(k)

    fieldnames = base_cols + sorted(extra_cols)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"Wrote summary: {summary_path}")
    print(f"Runs directory: {runs_dir}")
    print(f"Darts models discovered: {len(model_names)}")


if __name__ == "__main__":
    main()
