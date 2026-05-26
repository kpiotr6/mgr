"""Discrete MPC controller (grid search) using a Darts checkpoint model.

Assumptions / conventions (aligned with `config.py` and `outputs/min_max_data_preprocessed.csv`):

Controls (future covariates):
- clinker_1_feedrate in [2, 120]
- separator_speed in [280, 800]

Targets (multivariate series, 4 dims):
- return
- first_chamber_filling
- second_chamber_filling
- gran1_blain

The MPC:
- evaluates a grid (e.g. 5x5) of (clinker_1_feedrate, separator_speed)
- uses a control horizon of 12 steps split into 2 constant blocks of 6 steps
- forward horizon can be 15 / 30 / 60
- minimizes MAPE to setpoints, applies only the first action (receding horizon)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any
import sys

import numpy as np


TARGET_COLS = [
	"return",
	"first_chamber_filling",
	"second_chamber_filling",
	"gran1_blain",
]

CONTROL_COLS = [
	"clinker_1_feedrate",
	"separator_speed",
]


def _clip(value: float, lo: float, hi: float) -> float:
	return float(np.clip(value, lo, hi))


def _ensure_2d(a: np.ndarray) -> np.ndarray:
	if a.ndim == 1:
		return a.reshape(-1, 1)
	return a


@dataclass
class ScalerBundle:
	"""Persisted scalers matching the training pipeline (`darts` Scaler objects)."""

	target_scaler: Any
	future_covariates_scaler: Any | None
	meta: dict[str, Any]

	@staticmethod
	def load(path: str | Path) -> "ScalerBundle":
		path = Path(path)
		with path.open("rb") as f:
			obj = pickle.load(f)

		# Backwards-compatible loading (dict or ScalerBundle)
		if isinstance(obj, ScalerBundle):
			return obj

		if not isinstance(obj, dict):
			raise TypeError(f"Unsupported scalers format in {path}")

		return ScalerBundle(
			target_scaler=obj.get("target_scaler"),
			future_covariates_scaler=obj.get("covariates_scaler")
			or obj.get("future_covariates_scaler"),
			meta=obj.get("meta", {}),
		)

	def save(self, path: str | Path) -> None:
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		with path.open("wb") as f:
			pickle.dump(
				{
					"target_scaler": self.target_scaler,
					"covariates_scaler": self.future_covariates_scaler,
					"meta": self.meta,
				},
				f,
			)


def resolve_darts_model_class(model_class_name: str):
	"""Resolve a Darts model class by name.

	For your repo, `NeuralForecastModel` is commonly used.
	"""

	from darts import models as darts_models

	if not hasattr(darts_models, model_class_name):
		raise ValueError(
			f"Unknown Darts model class '{model_class_name}'. "
			f"Try e.g. 'NeuralForecastModel'."
		)

	model_class = getattr(darts_models, model_class_name)
	# Darts exposes optional models as a NotImportedModule placeholder when
	# extras are missing (e.g., NeuralForecastModel requires `neuralforecast`).
	try:
		from darts.utils.utils import NotImportedModule  # type: ignore
		is_placeholder = isinstance(model_class, NotImportedModule)
	except Exception:
		is_placeholder = model_class.__class__.__name__ == "NotImportedModule"

	if is_placeholder:
		raise ImportError(
			f"Darts model '{model_class_name}' is not available in this Python environment "
			"(optional dependency missing). "
			"Install the required extra (e.g. `neuralforecast`) in the same interpreter, "
			"or run using the project's virtualenv (e.g. `.venv/bin/python`)."
		)

	return model_class


def _timeseries_from_array(values: np.ndarray, columns: list[str], start: int = 0):
	"""Create a Darts TimeSeries from an array.

	Uses RangeIndex to avoid needing datetimes.
	"""

	import pandas as pd
	from darts import TimeSeries

	values = np.asarray(values, dtype=float)
	values = _ensure_2d(values)
	idx = pd.RangeIndex(start=start, stop=start + len(values), step=1)

	try:
		return TimeSeries.from_times_and_values(times=idx, values=values, columns=columns)
	except TypeError:
		# Older/newer Darts versions differ slightly.
		df = pd.DataFrame(values, columns=columns)
		df.insert(0, "t", idx)
		return TimeSeries.from_dataframe(df, time_col="t", value_cols=columns)


class DartsPredictor:
	def __init__(
		self,
		*,
		model_name: str,
		model_class_name: str = "NeuralForecastModel",
		model_dir: str | Path = "darts_logs",
		scalers_path: str | Path | None = None,
		target_cols: list[str] = TARGET_COLS,
		control_cols: list[str] = CONTROL_COLS,
	):
		self.model_name = model_name
		self.model_class_name = model_class_name
		self.model_dir = Path(model_dir)
		self.target_cols = list(target_cols)
		self.control_cols = list(control_cols)

		# Some saved checkpoints reference local modules like `model_definitions`
		# (because training prepends `model_training/` to sys.path). Make it
		# available here too so unpickling works.
		repo_root = Path(__file__).resolve().parents[1]
		model_training_dir = repo_root / "model_training"
		if str(model_training_dir) not in sys.path:
			sys.path.insert(0, str(model_training_dir))

		model_class = resolve_darts_model_class(model_class_name)
		try:
			if not hasattr(model_class, "load_from_checkpoint"):
				raise AttributeError("Model class has no load_from_checkpoint")
			self.model = model_class.load_from_checkpoint(model_name=model_name, best=True)
		except (AttributeError, FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError):
			# Fallback: load the serialized model artifact if present.
			model_tar = self.model_dir / model_name / "_model.pth.tar"
			if hasattr(model_class, "load") and model_tar.exists():
				self.model = model_class.load(str(model_tar))
			else:
				raise

		if scalers_path is None:
			scalers_path = self.model_dir / model_name / "scalers.pkl"

		scalers_path = Path(scalers_path)
		if not scalers_path.exists():
			raise FileNotFoundError(
				f"Missing scalers file: {scalers_path}. "
				"Create it (or retrain with scaler persistence enabled)."
			)

		self.scalers = ScalerBundle.load(scalers_path)

		meta_target_cols = self.scalers.meta.get("target_cols")
		if meta_target_cols and list(meta_target_cols) != self.target_cols:
			raise ValueError(
				f"Scaler bundle target_cols={meta_target_cols} does not match "
				f"requested target_cols={self.target_cols}"
			)

		meta_input_cols = self.scalers.meta.get("input_cols")
		if meta_input_cols and list(meta_input_cols) != self.control_cols:
			raise ValueError(
				f"Scaler bundle input_cols={meta_input_cols} does not match "
				f"requested control_cols={self.control_cols}. "
				"Train/fit scalers with exactly these control columns for MPC."
			)

		self.input_chunk_length = int(getattr(self.model, "input_chunk_length", 60))

	def predict(
		self,
		*,
		y_hist: np.ndarray,
		u_hist: np.ndarray,
		u_future: np.ndarray,
		n: int,
	) -> np.ndarray:
		"""Predict next n steps.

		Parameters
		----------
		y_hist: shape (T, 4)
			Past measured targets.
		u_hist: shape (T, 2)
			Past applied controls (aligned with y_hist timestamps).
		u_future: shape (n, 2)
			Planned controls for prediction window.
		n: int
			Forecast horizon.
		"""

		y_hist = np.asarray(y_hist, dtype=float)
		u_hist = np.asarray(u_hist, dtype=float)
		u_future = np.asarray(u_future, dtype=float)

		if y_hist.ndim != 2 or y_hist.shape[1] != len(self.target_cols):
			raise ValueError(f"y_hist must be (T,{len(self.target_cols)})")
		if u_hist.ndim != 2 or u_hist.shape[1] != len(self.control_cols):
			raise ValueError(f"u_hist must be (T,{len(self.control_cols)})")
		if u_future.ndim != 2 or u_future.shape[1] != len(self.control_cols):
			raise ValueError(f"u_future must be (n,{len(self.control_cols)})")
		if len(u_future) != n:
			raise ValueError("u_future length must equal n")

		from darts import TimeSeries

		y_ts: TimeSeries = _timeseries_from_array(y_hist, self.target_cols, start=0)

		u_all = np.concatenate([u_hist, u_future], axis=0)
		u_ts: TimeSeries = _timeseries_from_array(u_all, self.control_cols, start=0)

		y_scaled = self.scalers.target_scaler.transform(y_ts)
		u_scaled = (
			self.scalers.future_covariates_scaler.transform(u_ts)
			if self.scalers.future_covariates_scaler is not None
			else u_ts
		)

		# Most models used in this repo are in MODELS_WITH_FUTURE_COVARIATES.
		try:
			pred_scaled = self.model.predict(n=n, series=y_scaled, future_covariates=u_scaled, verbose=False)
		except TypeError:
			pred_scaled = self.model.predict(n=n, series=y_scaled, future_covariates=u_scaled)

		pred = self.scalers.target_scaler.inverse_transform(pred_scaled)
		return pred.values(copy=False)


class NaivePredictor:
	"""Naive baseline predictor: holds the last observed targets constant.

	This is useful as a baseline and for running MPC without a Darts checkpoint.
	It ignores controls and simply repeats the last row of `y_hist` for `n` steps.
	"""

	def __init__(
		self,
		*,
		target_cols: list[str] = TARGET_COLS,
		control_cols: list[str] = CONTROL_COLS,
		input_chunk_length: int = 60,
	):
		self.target_cols = list(target_cols)
		self.control_cols = list(control_cols)
		self.input_chunk_length = int(input_chunk_length)

	def predict(
		self,
		*,
		y_hist: np.ndarray,
		u_hist: np.ndarray,
		u_future: np.ndarray,
		n: int,
	) -> np.ndarray:
		y_hist = np.asarray(y_hist, dtype=float)
		u_hist = np.asarray(u_hist, dtype=float)
		u_future = np.asarray(u_future, dtype=float)

		if y_hist.ndim != 2 or y_hist.shape[1] != len(self.target_cols):
			raise ValueError(f"y_hist must be (T,{len(self.target_cols)})")
		if u_hist.ndim != 2 or u_hist.shape[1] != len(self.control_cols):
			raise ValueError(f"u_hist must be (T,{len(self.control_cols)})")
		if u_future.ndim != 2 or u_future.shape[1] != len(self.control_cols):
			raise ValueError(f"u_future must be (n,{len(self.control_cols)})")
		if len(u_future) != int(n):
			raise ValueError("u_future length must equal n")
		if len(y_hist) == 0:
			raise ValueError("y_hist must contain at least one row")

		last = y_hist[-1].reshape(1, -1)
		return np.repeat(last, repeats=int(n), axis=0)


@dataclass
class MPCConfig:
	forward_horizon: int = 30
	control_horizon: int = 12
	block_size: int = 6
	grid_n: int = 5
	clinker_1_feedrate_min: float = 2.0
	clinker_1_feedrate_max: float = 120.0
	separator_speed_min: float = 280.0
	separator_speed_max: float = 800.0
	weights: dict[str, float] | None = None


class DiscreteGridMPC:
	def __init__(
		self,
		*,
		predictor: DartsPredictor,
		setpoints: dict[str, float],
		config: MPCConfig | None = None,
	):
		self.predictor = predictor
		self.setpoints = {k: float(v) for k, v in setpoints.items()}
		self.cfg = config or MPCConfig()

		missing = [c for c in TARGET_COLS if c not in self.setpoints]
		if missing:
			raise ValueError(f"Missing setpoints for: {missing}")

		if self.cfg.control_horizon != self.cfg.block_size * 2:
			raise ValueError("This MPC expects exactly 2 blocks: control_horizon == 2*block_size")

		self._grid = self._make_grid()

		weights = self.cfg.weights or {}
		self._w = np.array([float(weights.get(col, 1.0)) for col in TARGET_COLS], dtype=float)

	def _make_grid(self) -> list[tuple[float, float]]:
		cfg = self.cfg
		fr = np.linspace(cfg.clinker_1_feedrate_min, cfg.clinker_1_feedrate_max, cfg.grid_n)
		ss = np.linspace(cfg.separator_speed_min, cfg.separator_speed_max, cfg.grid_n)
		return [(float(f), float(s)) for f in fr for s in ss]

	def _plan_from_two_blocks(
		self,
		*,
		u1: tuple[float, float],
		u2: tuple[float, float],
		horizon: int,
	) -> np.ndarray:
		cfg = self.cfg
		u = np.zeros((horizon, 2), dtype=float)
		u[: cfg.block_size, 0] = u1[0]
		u[: cfg.block_size, 1] = u1[1]
		u[cfg.block_size : min(horizon, cfg.control_horizon), 0] = u2[0]
		u[cfg.block_size : min(horizon, cfg.control_horizon), 1] = u2[1]
		if horizon > cfg.control_horizon:
			u[cfg.control_horizon :, 0] = u2[0]
			u[cfg.control_horizon :, 1] = u2[1]
		return u

	def choose_action(self, *, y_hist: np.ndarray, u_hist: np.ndarray) -> dict[str, float]:
		cfg = self.cfg
		n = int(cfg.forward_horizon)
		sp = np.array([self.setpoints[c] for c in TARGET_COLS], dtype=float).reshape(1, -1)
		u_last = np.asarray(u_hist, dtype=float)[-1]

		best_cost = float("inf")
		best_u0: tuple[float, float] | None = None
		best_du = float("inf")

		# Two-block search: 25 * 25 for 5x5 grid.
		for u1 in self._grid:
			for u2 in self._grid:
				u_future = self._plan_from_two_blocks(u1=u1, u2=u2, horizon=n)
				try:
					y_pred = self.predictor.predict(
						y_hist=y_hist,
						u_hist=u_hist,
						u_future=u_future,
						n=n,
					)
				except Exception:
					# If the model cannot evaluate this candidate (covariate mismatch etc), skip it.
					continue

				err = y_pred - sp
				# weighted MAPE over variables, then average over time
				# (epsilon avoids division by zero for near-zero setpoints)
				eps = 1e-6
				denom = np.maximum(np.abs(sp), eps)
				mape_t = np.mean((np.abs(err) / denom) * self._w.reshape(1, -1), axis=1)
				cost = float(np.mean(mape_t))

				if cost < best_cost:
					best_cost = cost
					best_u0 = u1
					best_du = float(np.sum((np.array(u1, dtype=float) - u_last) ** 2))
				elif best_u0 is not None and np.isfinite(cost) and np.isclose(cost, best_cost, rtol=1e-12, atol=1e-12):
					# Tie-break: prefer the action closest to the last applied control.
					# This makes naive/constant predictors behave as "do not change controls".
					du = float(np.sum((np.array(u1, dtype=float) - u_last) ** 2))
					if du < best_du:
						best_u0 = u1
						best_du = du

		if best_u0 is None:
			raise RuntimeError("MPC failed: no candidate control sequence could be evaluated")

		return {
			"clinker_1_feedrate": best_u0[0],
			"separator_speed": best_u0[1],
			"cost": best_cost,
		}

