"""
Discrete grid-search Model Predictive Control (MPC) for the cement mill process.

Loads one Darts forecasting model per target (e.g. "return", "gran1_blain",
"first_chamber_filling", "second_chamber_filling" - the model names/paths are
passed in by the caller, see `MPCController.__init__`). Each model was trained
by `model_training/train_models.py` under the "simple" run group, so it
forecasts a single target from:
  - the target's own recent history (lags),
  - a handful of *past* covariates (the other process targets), and
  - *future* covariates that are exactly the two control variables below.

Optimization strategy (as requested - a brute-force discretized grid search,
not a gradient/QP-based MPC):
  1. Discretize the two control variables on a grid (step of `control_step`).
  2. For every grid point, hold that control constant across the *entire*
     prediction horizon (a model may output more than one 5-minute step; we
     do not vary the control within the horizon - see `choose_action`).
  3. Predict every target with its own model for every grid point in one
     batched `model.predict(series=[...])` call per target (much faster than
     looping candidate-by-candidate).
  4. Score each candidate by the (weighted) mean absolute error - or mean
     absolute percentage error, see `error_metric` - between the predicted
     trajectory and the requested setpoint, summed over targets.
  5. Among candidates within `tie_break_tolerance` of the best cost, prefer
     the one closest to the currently-applied control, to avoid chattering
     between near-equally-good grid points.
  6. Return the best candidate. Only the *first* control step is meant to be
     applied (receding horizon) - the caller should call `choose_action`
     again once new measurements come in.
"""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from darts import TimeSeries

# 5 minutes per model step (see model_training/train_models.py: one row == one step).
STEP_MINUTES = 5.0

DEFAULT_CONTROL_BOUNDS: dict[str, tuple[float, float]] = {
    "separator_speed": (0.0, 750.0),
    "fresh_feed_setpoint": (40.0, 130.0),
}
DEFAULT_CONTROL_STEP = 10.0

# "mae": mean(|pred - setpoint|), same units/scale as the target itself.
# "mape": 100 * mean(|pred - setpoint| / |setpoint|) - scale-free, so it stops
# whichever target happens to have the largest raw magnitude (e.g. gran1_blain
# in the thousands) from dominating the combined multi-target cost.
VALID_ERROR_METRICS = ("mae", "mape")
DEFAULT_ERROR_METRIC = "mae"
MAPE_EPS = 1e-6

# Candidates within this fraction of the best cost are treated as "equally
# good" and tie-broken by proximity to the currently-applied control (see
# `choose_action`), instead of picking whichever happened to be first/lowest.
# This is what keeps the controller from chattering between two grid points
# that predict almost identical error.
DEFAULT_TIE_BREAK_TOLERANCE = 0.01
TIE_BREAK_ABS_TOL = 1e-3


def _resolve_model_class(model_dir_name: str):
    """Infer which Darts model class + loading strategy to use from the
    directory name, following the naming convention used by
    `model_training/train_models.py` (`{run_group}_{model_type}_I{in}_O{out}`).
    """
    from darts.models import LinearRegressionModel, NeuralForecastModel

    if "NeuralForecast" in model_dir_name:
        return NeuralForecastModel, "checkpoint"
    if "Chronos2" in model_dir_name:
        from darts.models import Chronos2Model

        return Chronos2Model, "serialized"
    if "LinearRegression" in model_dir_name:
        return LinearRegressionModel, "serialized"
    raise ValueError(
        f"Cannot infer a Darts model class from model directory name '{model_dir_name}'. "
        "Expected it to contain 'LinearRegression', 'NeuralForecast_*' or 'Chronos2'."
    )


def _load_target_model(model_dir_name: str, darts_logs_dir: Path):
    model_dir = darts_logs_dir / model_dir_name
    scalers_path = model_dir / "scalers.pkl"
    if not scalers_path.exists():
        raise FileNotFoundError(
            f"Missing {scalers_path}. Train this model first "
            "(see model_training/train_models.py --runs simple), which also persists "
            "the scalers MPC needs to reproduce the training-time normalization."
        )
    with scalers_path.open("rb") as f:
        bundle = pickle.load(f)

    model_class, kind = _resolve_model_class(model_dir_name)
    if kind == "checkpoint":
        model = model_class.load_from_checkpoint(
            model_name=model_dir_name, work_dir=str(darts_logs_dir), best=True
        )
    else:
        model_path = model_dir / "_model.pth.tar"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing serialized model file {model_path}.")
        model = model_class.load(str(model_path))

    return model, bundle


def _chunk_lengths(model) -> tuple[int, int]:
    """(input_chunk_length, output_chunk_length), across regression- and
    torch-based Darts models, which expose this differently."""
    if hasattr(model, "input_chunk_length"):
        in_len = int(model.input_chunk_length)
    else:
        in_len = len(model.lags["target"])
    return in_len, int(model.output_chunk_length)


def _series_from_frame(df: pd.DataFrame, columns: Sequence[str]) -> TimeSeries:
    """Build a Darts TimeSeries with a plain RangeIndex (0, 1, 2, ...) from a
    slice of the history frame. All series built this way share the same time
    axis convention, so target/past/future covariates line up automatically.
    """
    values = df[list(columns)].to_numpy(dtype=np.float32)
    return TimeSeries.from_values(values, columns=list(columns))


class MPCController:
    """Receding-horizon, discretized-grid-search MPC controller.

    Parameters
    ----------
    model_names:
        Mapping of target name -> Darts model directory name under
        `darts_logs_dir` (e.g. {"return": "simple_return_LinearRegression_I2_O2"}).
        One independent model per target, as requested.
    darts_logs_dir:
        Folder holding the trained model artifacts (default "darts_logs").
    control_bounds:
        Mapping of control variable name -> (min, max). Defaults to
        separator_speed in [0, 750] and fresh_feed_setpoint in [40, 130].
    control_step:
        Grid discretization step applied to every control variable.
    weights:
        Optional per-target weights used when summing the per-target error
        across targets (default 1.0 for every target). With `error_metric=
        "mae"`, targets have very different scales (e.g. gran1_blain ~
        thousands vs. filling % ~ 0-100), so if you optimize several targets
        together with MAE you likely want to rescale them, e.g. weight ~
        1 / typical_setpoint_magnitude - or use `error_metric="mape"`
        instead, which is scale-free.
    error_metric:
        "mae" (default) scores each candidate by mean absolute error against
        the setpoint, in the target's own units. "mape" scores by mean
        absolute *percentage* error instead (100 * |pred - setpoint| /
        |setpoint|), which puts every target on the same 0-100-ish scale
        regardless of its raw magnitude - useful when optimizing several
        targets together without hand-tuned `weights`.
    tie_break_tolerance:
        Candidates whose total cost is within this fraction of the best
        candidate's cost are considered ties, and among those the one
        closest to the currently-applied control (`self.last_control`,
        auto-updated after every `choose_action` call - see also
        `initial_control`) is preferred. This avoids chattering between two
        grid points that are predicted to perform almost identically. Set to
        0 to disable and always take the strict argmin.
    initial_control:
        Optional control values to seed `self.last_control` with, used only
        for tie-breaking on the very first `choose_action` call (before any
        control has actually been applied yet). If omitted, the first call
        has no tie-break reference and falls back to the strict argmin.
    """

    def __init__(
        self,
        model_names: Mapping[str, str],
        darts_logs_dir: str | Path = "darts_logs",
        control_bounds: Mapping[str, tuple[float, float]] | None = None,
        control_step: float = DEFAULT_CONTROL_STEP,
        weights: Mapping[str, float] | None = None,
        error_metric: str = DEFAULT_ERROR_METRIC,
        tie_break_tolerance: float = DEFAULT_TIE_BREAK_TOLERANCE,
        initial_control: Mapping[str, float] | None = None,
        history_maxlen: int = 200,
    ):
        if not model_names:
            raise ValueError("model_names must map at least one target to a model directory name.")
        error_metric = error_metric.lower()
        if error_metric not in VALID_ERROR_METRICS:
            raise ValueError(f"error_metric must be one of {VALID_ERROR_METRICS}, got '{error_metric}'.")

        self.darts_logs_dir = Path(darts_logs_dir)
        self.control_bounds = dict(control_bounds or DEFAULT_CONTROL_BOUNDS)
        self.control_vars = list(self.control_bounds.keys())
        self.control_step = float(control_step)
        self.weights = {t: float((weights or {}).get(t, 1.0)) for t in model_names}
        self.error_metric = error_metric
        self.tie_break_tolerance = float(tie_break_tolerance)
        self.last_control: dict[str, float] | None = (
            {v: float(initial_control[v]) for v in self.control_vars} if initial_control else None
        )

        self.models: dict[str, object] = {}
        self.bundles: dict[str, dict] = {}
        self.chunk_lengths: dict[str, tuple[int, int]] = {}

        for target, model_dir_name in model_names.items():
            model, bundle = _load_target_model(model_dir_name, self.darts_logs_dir)
            meta = bundle["meta"]
            unknown_inputs = set(meta.get("input_cols") or []) - set(self.control_vars)
            if unknown_inputs:
                raise ValueError(
                    f"Model for target '{target}' expects future covariates {meta['input_cols']}, "
                    f"but {sorted(unknown_inputs)} are not among the known control variables "
                    f"{self.control_vars}. This controller only knows how to plan for the "
                    "control variables themselves as future covariates."
                )
            self.models[target] = model
            self.bundles[target] = bundle
            self.chunk_lengths[target] = _chunk_lengths(model)

        # Columns that must be present in every history row: every target's own
        # column, every past-covariate column any model needs, and the controls.
        self.history_cols: list[str] = sorted(
            set(self.control_vars)
            | {c for b in self.bundles.values() for c in b["meta"]["target_cols"]}
            | {c for b in self.bundles.values() for c in (b["meta"].get("past_cols") or [])}
        )
        self.max_input_chunk_length = max(l[0] for l in self.chunk_lengths.values())

        self.history: deque[dict[str, float]] = deque(maxlen=history_maxlen)

    # ------------------------------------------------------------------
    def seed_history(self, rows: Sequence[Mapping[str, float]]) -> None:
        """Bulk-append initial history rows (oldest first), e.g. replayed from
        `data_preprocessed/*.csv` to warm up the controller before the first
        `choose_action` call."""
        for row in rows:
            self.update_history(row)

    def set_history(self, rows: Sequence[Mapping[str, float]]) -> None:
        """Replace the entire history buffer with `rows` (oldest first).

        Useful when the caller is running the plant faster than the models'
        native step size and re-derives the trailing "one row per model step"
        window from scratch every control cycle (e.g. mean-pooling a rolling
        window of raw 1-minute simulator samples into 5-minute-equivalent
        rows) rather than appending one true new row at a time.
        """
        self.history.clear()
        self.seed_history(rows)

    def update_history(self, measurement: Mapping[str, float]) -> None:
        """Append one new (5-minute) measurement. Must contain a value for
        every column in `self.history_cols`."""
        missing = [c for c in self.history_cols if c not in measurement]
        if missing:
            raise ValueError(f"Measurement is missing required columns: {missing}")
        self.history.append({c: float(measurement[c]) for c in self.history_cols})

    def has_enough_history(self) -> bool:
        return len(self.history) >= self.max_input_chunk_length

    # ------------------------------------------------------------------
    def _control_grid(self) -> np.ndarray:
        """All discretized combinations of the control variables, as an
        (n_candidates, n_control_vars) array (columns ordered like
        `self.control_vars`)."""
        axes = []
        for var in self.control_vars:
            lo, hi = self.control_bounds[var]
            n_steps = int(round((hi - lo) / self.control_step))
            axes.append(lo + self.control_step * np.arange(n_steps + 1))
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=-1)

    def _predict_target_over_grid(
        self, target: str, hist_df: pd.DataFrame, candidates: np.ndarray
    ) -> np.ndarray:
        """Batched prediction of `target` for every candidate control setting.
        Returns an (n_candidates, output_chunk_length) array of real-valued
        (inverse-scaled) predictions.
        """
        model = self.models[target]
        bundle = self.bundles[target]
        meta = bundle["meta"]
        in_len, out_len = self.chunk_lengths[target]

        target_cols = meta["target_cols"]
        past_cols = meta.get("past_cols") or []
        input_cols = meta.get("input_cols") or []

        hist_window = hist_df.iloc[-in_len:]
        target_ts = _series_from_frame(hist_window, target_cols)
        target_ts = bundle["target_scaler"].transform(target_ts) if bundle["target_scaler"] else target_ts

        past_ts = None
        if past_cols:
            past_ts = _series_from_frame(hist_window, past_cols)
            if bundle["past_covariates_scaler"]:
                past_ts = bundle["past_covariates_scaler"].transform(past_ts)

        series_list, past_list, future_list = [], [], []
        hist_future_raw = hist_window[input_cols].to_numpy(dtype=np.float32) if input_cols else None

        for candidate in candidates:
            control_values = dict(zip(self.control_vars, candidate))
            series_list.append(target_ts)
            past_list.append(past_ts)
            if input_cols:
                future_row = np.array([control_values[c] for c in input_cols], dtype=np.float32)
                future_values = np.vstack([hist_future_raw, np.tile(future_row, (out_len, 1))])
                future_ts = TimeSeries.from_values(future_values, columns=input_cols)
                if bundle["covariates_scaler"]:
                    future_ts = bundle["covariates_scaler"].transform(future_ts)
                future_list.append(future_ts)

        predict_kwargs = dict(n=out_len, series=series_list, verbose=False)
        if past_cols:
            predict_kwargs["past_covariates"] = past_list
        if input_cols:
            predict_kwargs["future_covariates"] = future_list

        preds_scaled = model.predict(**predict_kwargs)
        target_idx = target_cols.index(target) if target in target_cols else 0

        out = np.empty((len(candidates), out_len))
        for i, pred_scaled in enumerate(preds_scaled):
            pred = bundle["target_scaler"].inverse_transform(pred_scaled) if bundle["target_scaler"] else pred_scaled
            out[i] = pred.values(copy=False)[:, target_idx]
        return out

    def _candidate_errors(self, preds: np.ndarray, setpoint: float) -> np.ndarray:
        """Per-candidate error against `setpoint`, averaged over the
        prediction horizon, using `self.error_metric`."""
        abs_err = np.abs(preds - setpoint)
        if self.error_metric == "mape":
            denom = max(abs(setpoint), MAPE_EPS)
            return np.mean(abs_err / denom, axis=1) * 100.0
        return np.mean(abs_err, axis=1)

    def _break_ties(self, total_cost: np.ndarray, candidates: np.ndarray, best_idx: int) -> int:
        """Among candidates within `self.tie_break_tolerance` (relative) of
        the best cost, return the index closest to `self.last_control`
        (control-range-normalized L1 distance), instead of the strict argmin.
        """
        tol = max(abs(total_cost[best_idx]) * self.tie_break_tolerance, TIE_BREAK_ABS_TOL)
        near_idx = np.nonzero(total_cost <= total_cost[best_idx] + tol)[0]
        if len(near_idx) <= 1:
            return best_idx

        ranges = np.array([self.control_bounds[v][1] - self.control_bounds[v][0] for v in self.control_vars])
        last = np.array([self.last_control.get(v, candidates[best_idx, j]) for j, v in enumerate(self.control_vars)])
        dists = np.sum(np.abs(candidates[near_idx] - last) / ranges, axis=1)
        return int(near_idx[np.argmin(dists)])

    # ------------------------------------------------------------------
    def choose_action(self, setpoints: Mapping[str, float]) -> dict:
        """Evaluate every discretized control combination and return the one
        minimizing the (weighted) sum of per-target error against
        `setpoints`, using `self.error_metric` ("mae" or "mape").

        Only targets present in both `setpoints` and the models this
        controller was built with are optimized against; others are ignored.

        Returns a dict with:
          - "control": {control_var: value} - apply this control for the next
            step only (receding horizon - call `choose_action` again once new
            measurements arrive via `update_history`).
          - "cost": the winning candidate's weighted total error.
          - "predictions": {target: np.ndarray of shape (output_chunk_length,)}
            predicted trajectory for the winning candidate.
          - "error": {target: float} per-target error (in `self.error_metric`
            units) for the winning candidate.
        """
        if not self.has_enough_history():
            raise RuntimeError(
                f"Need at least {self.max_input_chunk_length} history rows before "
                f"choosing an action, have {len(self.history)}. Call update_history()/seed_history() first."
            )
        targets = [t for t in self.models if t in setpoints]
        if not targets:
            raise ValueError(f"None of the setpoints {list(setpoints)} match a loaded target model.")

        hist_df = pd.DataFrame(list(self.history))
        candidates = self._control_grid()

        total_cost = np.zeros(len(candidates))
        predictions: dict[str, np.ndarray] = {}
        errors: dict[str, np.ndarray] = {}
        for target in targets:
            preds = self._predict_target_over_grid(target, hist_df, candidates)
            predictions[target] = preds
            error = self._candidate_errors(preds, setpoints[target])
            errors[target] = error
            total_cost += self.weights.get(target, 1.0) * error

        best_idx = int(np.argmin(total_cost))

        if self.tie_break_tolerance > 0 and self.last_control is not None:
            best_idx = self._break_ties(total_cost, candidates, best_idx)

        best_control = {var: float(candidates[best_idx, j]) for j, var in enumerate(self.control_vars)}
        self.last_control = best_control

        return {
            "control": best_control,
            "cost": float(total_cost[best_idx]),
            "predictions": {t: predictions[t][best_idx] for t in targets},
            "error": {t: float(errors[t][best_idx]) for t in targets},
        }
