# `mpc_controller.py` — Controller Description

This document explains how `MPCController` (in `mpc2/mpc_controller.py`) works.

It is **not** a classical QP/gradient-based MPC. It is a *receding-horizon,
brute-force grid-search* MPC: the control space is discretized, every candidate
is simulated forward with trained [Darts](https://unit8co.github.io/darts/)
forecasting models, and the candidate with the lowest cost is applied. The
learned models play the role that a linearized state-space model plays in a
textbook MPC — they are the internal prediction model.

```
                ┌──────────────────────────────────────────────┐
                │  MPCController.choose_action(setpoints)       │
                │                                              │
 history  ────► │  1. _control_grid()      → 760 candidates     │
 (deque of      │  2. _predict_target_over_grid()  (per target) │ ────► control
  5-min rows)   │  3. _reference_trajectory() + _candidate_...  │       {sep_speed,
                │  4. _move_penalty()                          │        fresh_feed}
                │  5. argmin + _break_ties()                   │
                └──────────────────────────────────────────────┘
                                    ▲                  │
                                    └── update_history ┘  (receding horizon)
```

---

## 1. The prediction model: one Darts model per target

The controller is constructed with a mapping *target name → model directory
name* under `darts_logs/`, e.g. (from `run_mpc_on_simulator.py`):

```python
MODEL_NAMES = {
    "return":                  "simple_return_LinearRegression_I6_O3",
    "first_chamber_filling":   "simple_first_chamber_filling_NeuralForecast_TSMixer_I2_O3",
    "second_chamber_filling":  "simple_second_chamber_filling_NeuralForecast_TSMixer_I2_O3",
    "gran1_blain":             "simple_gran1_blain_LinearRegression_I6_O3",
}
```

Each target has its **own independent model** — there is no single joint
multi-output plant model. The models were trained by
`model_training/train_models.py` under the `simple` run group, and the naming
convention `{run_group}_{target}_{model_type}_I{in}_O{out}` is what
`_resolve_model_class` parses to decide *how* to load the artifact:

| name contains       | class                  | loading strategy               |
|---------------------|------------------------|--------------------------------|
| `NeuralForecast`    | `NeuralForecastModel`  | `load_from_checkpoint(best=True)` |
| `Chronos2`          | `Chronos2Model`        | serialized `_model.pth.tar`    |
| `LinearRegression`  | `LinearRegressionModel`| serialized `_model.pth.tar`    |

Alongside the model, `_load_target_model` unpickles `scalers.pkl`, which carries
the **training-time normalizers** (`target_scaler`, `past_covariates_scaler`,
`covariates_scaler`) plus a `meta` dict listing which columns played which role.
Without that bundle the controller cannot reproduce the normalization the model
was trained under, so a missing `scalers.pkl` is a hard `FileNotFoundError` with
a pointer to retrain.

### The three input groups

Every model consumes:

1. **target lags** — the target's own recent history (`input_chunk_length` rows);
2. **past covariates** — the other process variables over the same window;
3. **future covariates** — *exactly the two control variables*.

Group 3 is what makes these forecasters usable for control at all: the future
covariates are the decision variables. `__init__` validates this invariant
strictly (`mpc_controller.py:283`) — if any model expects a future covariate
that is not a declared control variable, construction fails, because the
controller has no way to plan a value for it.

### Derived bookkeeping

```python
self.history_cols          # union of controls ∪ all target_cols ∪ all past_cols
self.max_input_chunk_length  # max in_len across models — the warm-up requirement
self.chunk_lengths[target]   # (input_chunk_length, output_chunk_length)
```

`_chunk_lengths` normalizes over the fact that regression models expose their
lag structure as `model.lags["target"]` while torch models expose
`model.input_chunk_length`.

---

## 2. History buffer

The controller keeps a `deque` of rows, **one row per 5-minute model step**
(`STEP_MINUTES = 5.0`), each row containing a value for every column in
`history_cols`. Three entry points:

- `update_history(measurement)` — append one new step; raises if any required
  column is missing;
- `seed_history(rows)` — bulk-append (oldest first), to warm up from replayed
  CSV data;
- `set_history(rows)` — *replace* the whole buffer.

`set_history` exists for the rate-mismatch case in `run_mpc_on_simulator.py`:
the simulator and the decision loop run at **1 minute**, but the models step at
**5 minutes**. Instead of waiting five minutes between decisions, the runner
rebuilds the entire pooled window every minute — a rolling window of trailing
raw 1-minute samples, mean-pooled into 5-minute-equivalent blocks. The model
always sees its native step size while the controller replans every minute.

`has_enough_history()` gates `choose_action` on `max_input_chunk_length`.

---

## 3. Candidate generation — discretizing the control space

```python
DEFAULT_CONTROL_BOUNDS = {
    "separator_speed":     (0.0, 750.0),
    "fresh_feed_setpoint": (40.0, 130.0),
}
DEFAULT_CONTROL_STEP = 10.0
```

`_control_grid` (`mpc_controller.py:338`) builds one axis per control variable
by stepping from `lo` to `hi` in increments of `control_step`, then takes the
full Cartesian product via `np.meshgrid`:

- `separator_speed`: 76 values (0, 10, …, 750)
- `fresh_feed_setpoint`: 10 values (40, 50, …, 130)
- → **760 candidates**, as an `(760, 2)` array with columns ordered like
  `self.control_vars`.

Two consequences worth stating plainly:

- **Constraints are structural.** The grid never leaves the bounds, so there is
  no constraint handling and no infeasibility to recover from — unlike a QP MPC.
- **Each candidate is a single constant move.** It is held constant across the
  *entire* prediction horizon. The controller does not plan a varying control
  sequence over the horizon; it plans one value and relies on replanning.

---

## 4. Batched prediction over the grid

`_predict_target_over_grid` (`mpc_controller.py:350`) is where nearly all the
compute goes. For one target:

```python
hist_window = hist_df.iloc[-in_len:]
target_ts   = scaler.transform(_series_from_frame(hist_window, target_cols))
past_ts     = scaler.transform(_series_from_frame(hist_window, past_cols))
```

The target series and past-covariate series are **identical for every
candidate** — they are historical fact, unaffected by the decision — so they are
built once and the *same objects* are appended 760 times into the batch list.
Only the future covariates differ per candidate:

```python
future_values = np.vstack([hist_future_raw,              # historical controls
                           np.tile(future_row, (out_len, 1))])  # candidate, held constant
```

i.e. the past control values followed by the candidate tiled across the whole
output horizon. Each is scaled with the training-time `covariates_scaler`.

Then **one single batched call per target**:

```python
preds_scaled = model.predict(n=out_len, series=series_list,
                             past_covariates=past_list,
                             future_covariates=future_list, verbose=False)
```

Batching rather than looping candidate-by-candidate is the main reason the
exhaustive search is tractable at all — 4 `predict` calls per control cycle
instead of 3040. Predictions are inverse-transformed back to real units and
returned as an `(n_candidates, out_len)` array.

All series are built with a plain `RangeIndex` via `TimeSeries.from_values`
(`_series_from_frame`), so target / past / future axes line up automatically
without any date arithmetic.

> **Side note — log noise.** PyTorch Lightning re-emits its "GPU/TPU/HPU
> available" and "LOCAL_RANK" banners on *every* `predict()` call, which the
> grid search makes intolerable. The module silences them at import
> (`mpc_controller.py:61-66`), and deliberately does so *after* importing
> `lightning_fabric` / `pytorch_lightning`, since both reset their own logger
> level to INFO inside their `__init__.py`.

---

## 5. The cost function

### 5.1 Reference trajectory instead of a flat setpoint

`_reference_trajectory` (`mpc_controller.py:407`) does not ask the plant to hit
the setpoint immediately. It builds a per-step reference that **geometrically
approaches** the setpoint `S` from the current measurement `C`: each step is the
midpoint between the previous reference value and `S`, with the final horizon
step being `S` itself. For `out_len = 3`:

```
[ (C+S)/2 ,  ((C+S)/2 + S)/2 ,  S ]
```

This is effectively a first-order reference governor. It softens the near-term
demand — the controller is only asked to *close half the gap* on the first step —
which by itself substantially reduces aggressive control action, before any
explicit move penalty is involved.

### 5.2 Per-target error

`_candidate_errors` averages the absolute deviation from that reference over the
horizon, under one of two metrics:

- `"mae"` — `mean(|pred − ref|)`, in the target's own units;
- `"mape"` — `100 · mean(|pred − ref| / |ref|)`, scale-free;
- `"itae"` — `Σ_k t_k · |pred[k] − ref[k]| / Σ_k t_k` with `t_k = k · STEP_MINUTES`,
  the discrete integral of time-weighted absolute error.

The MAE/MAPE distinction matters because the targets have wildly different
magnitudes: `gran1_blain ≈ 5500` versus filling percentages in `0–100`. Under
MAE, Blaine error dominates the summed cost unless compensated. The runner uses
`"mae"` and compensates with weights (`gran1_blain: 0.02`); `"mape"` is the
alternative that needs no hand-tuning.

ITAE is the orthogonal choice: it keeps MAE's units but stops treating every
horizon step as equally important. A deviation at step `N` costs `N ×` what the
same deviation costs at step 1, so the optimizer prefers a candidate that
*settles* over one that merely looks good on the first step and drifts
afterwards — the standard argument for ITAE as a tuning criterion. The
implementation (`_time_weights`, `mpc_controller.py:424`) normalizes the time
weights to average 1 rather than using raw `t_k`, which leaves only the relative
`1 : 2 : … : N` emphasis: a constant error across the horizon scores identically
under `"mae"` and `"itae"`, so `weights`, `move_penalty` and
`tie_break_tolerance` transfer between the two without retuning.

Note that ITAE interacts with the reference trajectory of §5.1: the reference
already ramps towards the setpoint and only demands `S` itself at step `N`,
which is precisely the step ITAE weights most heavily. The two therefore
reinforce each other — soft early demand, strictly enforced endpoint.

### 5.3 Move suppression

`_move_penalty` (`mpc_controller.py:453`) penalizes distance from the
currently-applied control, normalized by each variable's bounds range so the
weights are comparable across variables of different physical scale:

```
penalty(cand) = Σ_v  move_penalty[v] · |cand[v] − last_control[v]| / range[v]
```

It returns zeros when there is no `last_control` yet (first-ever call) or every
weight is 0. This is a **genuine cost trade-off**: a candidate predicting
slightly lower tracking error can lose to one that stays closer to the current
control. That distinguishes it from the tie-break below, which only
disambiguates candidates that already score the same.

### 5.4 Total

```
total_cost(cand) = Σ_targets  weight[t] · error_metric(pred[t](cand), reference[t])
                 + Σ_controls move_penalty[v] · |cand[v] − last[v]| / range[v]
```

Only targets present in **both** `setpoints` and the loaded models are optimized
against; anything else is silently ignored.

---

## 6. Selection and tie-breaking

```python
best_idx = int(np.argmin(total_cost))
if self.tie_break_tolerance > 0 and self.last_control is not None:
    best_idx = self._break_ties(total_cost, candidates, best_idx)
```

`_break_ties` (`mpc_controller.py:467`) collects every candidate within

```python
tol = max(abs(total_cost[best_idx]) * self.tie_break_tolerance, TIE_BREAK_ABS_TOL)
```

of the best cost (relative tolerance with an absolute floor of `1e-3`, so it
still behaves sensibly when the best cost is near zero), and among those picks
the one with the smallest range-normalized L1 distance to `last_control`.

This is the anti-chattering mechanism. Because the runner replans **every
simulated minute** while the underlying models only step every five, consecutive
cycles see almost the same history and produce almost the same costs — without
tie-breaking the controller would flip between two adjacent grid points
indefinitely. With it, a near-optimal candidate that happens to *be* the current
control wins.

The winner is stored in `self.last_control`, which becomes the reference for
both the move penalty and the tie-break on the next cycle.

### Return value

```python
{
  "control":     {control_var: value},          # apply this for the next step only
  "cost":        float,                          # winning total cost, incl. move penalty
  "predictions": {target: np.ndarray(out_len)},  # winner's predicted trajectory
  "error":       {target: float},                # per-target error, in error_metric units
}
```

---

## 7. Receding horizon

Only the **first** step of the winning control is meant to be applied. The
caller applies it, advances the plant, pushes fresh measurements in via
`update_history` / `set_history`, and calls `choose_action` again. The horizon
slides forward one step per cycle — the standard receding-horizon principle,
which is what turns an open-loop optimizer into feedback control.

---

## 8. Properties and trade-offs

**What the grid search buys:**

- **Global optimum on the grid.** No local minima, no initialization
  sensitivity, no convergence tuning.
- **No gradients required**, so the internal model can be *anything* — a linear
  regressor, a TSMixer neural net, a foundation model like Chronos2 — mixed
  freely across targets in the same controller.
- **Constraints for free**, as described in §3.

**What it costs:**

- Complexity is `O(grid_points × n_targets)` model evaluations, and
  `grid_points` grows **exponentially in the number of control variables**.
  Two controls at step 10 → 760 candidates, fine. A third control would make it
  impractical without coarsening the grid or moving to a smarter search.
- **Resolution is quantized** by `control_step`; the true optimum between grid
  points is unreachable.
- **No intra-horizon control planning.** The controller cannot express "move
  hard now, back off later" — every candidate is a constant.
- **No explicit rate limit.** Nothing forbids a jump from one end of the range
  to the other in a single step. Smoothness comes entirely from the combination
  of the softened reference trajectory (§5.1), the move penalty (§5.3), and the
  tie-break (§6).
- **Model mismatch is unmodelled.** There is no disturbance estimator or
  integral action; steady-state offset from model bias is corrected only
  indirectly, through the reference trajectory being recomputed from the latest
  *measured* value every cycle.

---

## 9. Tuning knobs at a glance

| Parameter | Effect |
|---|---|
| `control_step` | Grid resolution vs. compute. Halving it roughly quadruples cost for 2 controls. |
| `control_bounds` | Hard actuator limits; also the normalizer for both penalty terms. |
| `weights` | Per-target priority in the summed cost. Under `"mae"`, also the scale equalizer. |
| `error_metric` | `"mae"` (target units, needs weights) vs. `"mape"` (scale-free) vs. `"itae"` (target units, weights late-horizon error more). |
| `move_penalty` | Reluctance to move the controls. Competes directly with tracking error. |
| `tie_break_tolerance` | Width of the "equally good" band; `0` disables and takes the strict argmin. |
| `initial_control` | Seeds `last_control` so the very first call already has a tie-break/penalty reference. |
| `history_maxlen` | Ring-buffer depth; only the trailing `max_input_chunk_length` rows are ever used. |
