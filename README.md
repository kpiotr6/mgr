# Cement mill: forecasting models and grid-search MPC

Data-driven modelling of an industrial cement mill and a Model Predictive
Controller built on top of the resulting forecasters.

This project is open source — the full source code is freely available to read,
run, modify and redistribute. See [License](#license).

The pipeline has three stages, each one script:

```
data/*.csv                         raw exported mill sessions
   |
   |  data_functionalities/preprocess_data.py     normalise / resample / clean
   v
data_preprocessed/*.csv
   |
   |  model_training/train_models.py              train + evaluate a model zoo
   |  (models come from model_training/model_definitions.py)
   v
darts_logs/<model_name>/           checkpoints + scalers.pkl
outputs/                           metrics + prediction dumps
   |
   |  mpc2/run_mpc_on_simulator.py                closed-loop MPC on a physics sim
   v
mpc2/mpc_state_per_minute*.csv, mpc_closed_loop_iae*.csv, fig_mpc_closed_loop*.png
```

`config.py` is the single source of truth for column names shared by all three
stages: `TARGET_COLS` (`return`, `first_chamber_filling`,
`second_chamber_filling`, `gran1_blain`), `INPUT_COLS` (the manipulable
setpoints), `PAST_COLS` (measured-only covariates), `SESSION_COL`
(`session_index`), `TIME_COL` (`sequence_index`), plus the per-run covariate
selections `PER_TARGET_CONFIG` / `SIMPLE_MODEL_CONFIG` and prediction `BOUNDS`.

---

## 1. Preprocessing — `data_functionalities/preprocess_data.py`

Reads every `*.csv` in the input directory and writes a cleaned copy of each to
the output directory, keeping only `session_index`, `sequence_index` and the
union of `INPUT_COLS + PAST_COLS + TARGET_COLS`. A file missing any of those
columns is a hard error.

Per file, the work is:

1. Optional value cleanup — clip negatives to 0 (`--zero-negatives`), replace
   exact zeros in the target columns with `0.1` (`--replace-target-zeros`, useful
   when a later log transform or MAPE would otherwise blow up).
2. Split into sessions by `session_index`.
3. Optional truncation at the first row where `gran1_blain` is `0`/`NaN`
   (`--check-gran-empty`) — the granulometer only reports intermittently, so this
   cuts a session at the point its quality measurement dies.
4. Normalise each session: sort by `sequence_index`, rebase it to start at 1,
   reindex onto a gap-free integer range, and forward-fill the holes.
5. Optional non-overlapping subsampling (`--subsample-window`, `--subsample-method`)
   — the window is averaged (or median-aggregated) into a single row and
   `sequence_index` is renumbered. This is what sets the model time step: the
   trained models and the MPC assume **one row == 5 minutes**.
6. Optional splitting into fixed-length chunks (`--chunk-size`), each becoming a
   new session with its own id; leftovers shorter than the chunk are discarded.
7. Optional moving averages (`--ma-window`), added as `<col>_ma_<w>` columns
   computed within each session.

Statistics are printed per file and globally: rows in → rows out, sequence count,
and min / Q1 / median / Q3 / max session length. Those length quantiles are what
you check before picking `input_chunk_lengths` in training — sessions shorter than
`max(input) + max(output)` are dropped there.

```bash
python data_functionalities/preprocess_data.py \
    --input-dir data --output-dir data_preprocessed \
    --subsample-window 5 --check-gran-empty --zero-negatives
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input-dir` | str | `data` | Folder with raw CSV files. |
| `--output-dir` | str | `data_preprocessed` | Folder for processed CSV files. |
| `--ma-window` | int | `0` | Rolling window for per-session moving averages. `0` disables. |
| `--subsample-window` | int | `0` | Non-overlapping window averaged into one row. `0` disables. |
| `--subsample-method` | `mean`/`median` | `mean` | Aggregation used by subsampling. |
| `--check-gran-empty` | flag | off | Truncate a session at the first `gran1_blain` that is `0` or `NaN`. |
| `--chunk-size` | int | `0` | Cut sessions into non-overlapping chunks of exactly this length; shorter leftovers are dropped. `0` disables. |
| `--zero-negatives` | flag | off | Clip all negative numeric values to `0`. |
| `--replace-target-zeros` | flag | off | Replace exact `0` in `TARGET_COLS` with `0.1`. |

---

## 2. Model definitions — `model_training/model_definitions.py`

`get_models(input_chunk_length, output_chunk_length, input_cols, past_cols, run_group, use_builtin_scalers)`
instantiates every candidate model for one `(I, O)` configuration and returns them
as a `{name: model}` dict. Currently:

| Name | Backend | Notes |
| :--- | :--- | :--- |
| `NaiveLastValue` | `LastValueRepeater` (local) | Repeats the last observed value `n` times — the baseline every other model has to beat. |
| `LinearRegression` | `darts.LinearRegressionModel` | Lags = `input_chunk_length`; past-covariate lags and future-covariate lags `(I, O)` are added only when those covariate sets are non-empty. |
| `Chronos2` | `darts.Chronos2Model` | Foundation model (`autogluon/chronos-2-small`), registered only if the installed Darts exposes it; trained with `epochs=0`, i.e. zero-shot. |
| `NeuralForecast_BiTCN` | `NeuralForecastModel("BiTCN")` | `hidden_size=256`. |
| `NeuralForecast_TSMixer` | `NeuralForecastModel("TSMixerx")` | `n_block=6`, `ff_dim=128`, `dropout=0.1`. |
| `NeuralForecast_Nhits` | `NeuralForecastModel("NHITS")` | 3×3 blocks of `[1024,1024,1024]` MLP units, pooling/downsampling `[8,4,1]`, LeakyReLU. |
| `NeuralForecast_XLinear` | `NeuralForecastModel("XLinear")` | `hidden_size=512`, `temporal_ff=512`, `channel_ff=256`, lr `1e-2`. |
| `NeuralForecast_TFT` | `NeuralForecastModel("TFT")` | `hidden_size=512`, `n_head=16`, batch size 512. |

All neural models share: `RMSE` loss, 30 epochs, batch size 1024 (except TFT),
Adam at lr `1e-3` with `weight_decay=1e-4`, `ReduceLROnPlateau` (factor 0.5),
`save_checkpoints=True` and `force_reset=True`, so training always starts clean
and the best epoch is recoverable afterwards.

Three module-level lists drive how `train_models.py` treats each model, and they
are also what `--model-groups` selects on:

- `MODELS_WITH_FUTURE_COVARIATES` — accepts both `past_covariates` and
  `future_covariates` in `fit`/`predict`.
- `MODELS_FUTURE_COVARIATES_ONLY` — future covariates only.
- `CHECKPOINT_MODELS` — trains through PyTorch Lightning and is reloaded from
  its best checkpoint before evaluation.
- `NAIVE_MODELS` — refitted on each test window instead of trained globally.

Two supporting pieces live here as well:

- **`LossPlotCallback`** — a Lightning callback that scrapes train/val loss out of
  `trainer.callback_metrics` (tolerating the various `*_epoch` / `*_step` naming
  variants), and re-renders an annotated loss curve to
  `outputs/logs/I<I>_O<O>/<run_group>/<MODEL>/plots/<MODEL>_loss.png` after every
  validation epoch. A CSV logger writes to the same directory tree.
- **`neuralforecast.common._base_model.BaseModel.configure_optimizers`** —
  NeuralForecast returns a `ReduceLROnPlateau` scheduler without the `monitor` key
  Lightning requires, which raises `MisconfigurationException`. The patch injects
  `monitor="ptl/val_loss"`. It is applied at import time and silently skipped if
  the internal module ever moves.

---

## 3. Training and evaluation — `model_training/train_models.py`

Sweeps `input_chunk_lengths = [2, 4, 6, 8, 10]` × `output_chunk_lengths = [1, 2, 3]`
and, for each `(I, O)`, trains every selected model from the zoo. Data is read from
`data_preprocessed/`.

**Run groups** (`--runs`) decide which targets/covariates each training run uses:

| Run | Targets | Covariates | Artifact prefix |
| :--- | :--- | :--- | :--- |
| `all_targets` | all of `TARGET_COLS` at once (multivariate) | `INPUT_COLS` / `PAST_COLS` | `all_targets` |
| `per_target` | one run per target | that target's entry in `PER_TARGET_CONFIG` | `per_target_<target>` |
| `simple` | one run per target | that target's entry in `SIMPLE_MODEL_CONFIG` — only the other targets as past covariates and the two controls as future covariates | `simple_<target>` |
| `all` | all three of the above | | |

The `simple` group is the one the MPC consumes: its future covariates are exactly
`fresh_feed_setpoint` and `separator_speed`, which is what makes a model usable as
a control-response predictor.

**Pipeline per run:**

1. `load_data` splits each CSV at every `session_index` change and builds one
   `TimeSeries` triple (targets / future covariates / past covariates) per block,
   dropping blocks shorter than `max(I) + max(O)`.
2. Optional transforms, in order: exponential smoothing (`--exp-smoothing-alpha`),
   log transform (`--use-log-transform`), linear detrending (`--use-detrend`).
   The pre-transform series are kept aside so metrics can be computed against raw
   ground truth (or against the smoothed series with `--eval-on-smoothed`).
3. Split 70 / 15 / 15 — across the list of blocks by default, or *within* each
   session with `--split-per-session`. `--shuffle` shuffles blocks (seed 42) first.
4. A **global length filter** on `max(I) + max(O)` is applied to all three splits
   before any training, so every `(I, O)` configuration and every model is scored
   on the exact same dataset.
5. Scaling: a global-fit Darts `Scaler` for targets and each covariate set
   (optionally preceded by a Yeo-Johnson `PowerTransformer` via `--use-box-cox`),
   or the models' own internal scalers with `--use-builtin-scalers`.
6. Fit, with `val_series` passed for early stopping / checkpoint selection.
   Checkpoint models are reloaded with `load_from_checkpoint(best=True)` before
   testing; `LinearRegression` and `Chronos2` are serialised to
   `darts_logs/<model_name>/_model.pth.tar`.
7. Evaluate with a rolling origin: for every test block, every forecast origin from
   `max(I)` to the end, forecast `O` steps and invert the whole transform stack
   (scaler → box-cox → detrend → log), optionally clipping to `config.BOUNDS`
   (`--clip-predictions`). MAE / RMSE / MAPE are accumulated overall and per target
   column.

**Artifacts:**

- `darts_logs/<artifact_group>_<model>_I<I>_O<O>/` — checkpoint or `_model.pth.tar`,
  plus `scalers.pkl` (both scalers, both power transformers, and a `meta` block
  recording the exact target/input/past column lists). The MPC loads exactly this.
- `outputs/evaluation_metrics_<run_tag>.csv` — one row per (run, I, O, model), with
  `MAE`, `RMSE`, `MAPE` and per-target `MAE_<col>` / `RMSE_<col>` / `MAPE_<col>`.
- `outputs/pred_<run_tag>_<model>_I<I>_O<O>.csv` and
  `outputs/true_values_<run_tag>_I<I>_O<O>.csv` — every forecast window with its
  `block_idx` and `fcst_origin`, for post-hoc plotting.
- `outputs/logs/…` — Lightning CSV logs and loss plots.

```bash
python model_training/train_models.py --runs simple --split-per-session \
    --disable-mape --clip-predictions
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--runs` | one or more of `all`, `all_targets`, `per_target`, `simple` | `all` | Which training runs to execute. |
| `--models` | comma-separated str | — | Only train these model names. |
| `--exclude-models` | comma-separated str | — | Skip these model names. |
| `--model-groups` | one or more of `naive`, `checkpoint`, `future_covariates`, `future_only`, `other` | — | Only train models in these groups. |
| `--use-detrend` | flag | off | Linear detrend before training, reversed after prediction. |
| `--use-log-transform` | flag | off | Log transform before training, reversed after prediction. |
| `--use-box-cox` | flag | off | Yeo-Johnson `PowerTransformer`, reversed after prediction. |
| `--exp-smoothing-alpha` | float | — | EWM smoothing applied to all variables before training. |
| `--eval-on-smoothed` | flag | off | Score against the smoothed series instead of the raw one. |
| `--ma-window` | int | `0` | Extend `INPUT_COLS`/`PAST_COLS` (and both per-target configs) with `_ma_<w>` columns. Target MAs go to past covariates only, to avoid leakage. |
| `--shuffle` | flag | off | Shuffle blocks (seed 42) before splitting. |
| `--split-per-session` | flag | off | Split each session 70/15/15 in time instead of splitting the block list. |
| `--use-builtin-scalers` | flag | off | Use the models' own scalers instead of Darts `Scaler`; disables `scalers.pkl`. |
| `--disable-mape` | flag | off | Skip MAPE (avoids division-by-zero on near-zero targets); model selection then tracks MAE. |
| `--clip-predictions` | flag | off | Clip test predictions to `config.BOUNDS`. |

`run.sh` is the SLURM batch wrapper used on the GPU cluster (48h, one A100).

---

## 4. Closed-loop MPC — `mpc2/run_mpc_on_simulator.py`

Runs `mpc2/mpc_controller.MPCController` in closed loop against the physics-based
`mpc2/cement_mill_sim.CementMillSimulator`, and plots the result with setpoints
overlaid.

```bash
.venv/bin/python -m mpc2.run_mpc_on_simulator          # per-target sweep (default)
.venv/bin/python -m mpc2.run_mpc_on_simulator --single # one run, full setpoint vector
```

**How the controller works.** It loads one `simple`-group Darts model per target
from `darts_logs/` (names in `MODEL_NAMES`), then at each decision:
discretises the two controls onto a grid of step `CONTROL_STEPS`, holds each
candidate constant across the whole horizon, predicts every target for every grid
point in one batched `predict` call, scores each candidate against a reference
trajectory that geometrically approaches the setpoint, adds a move-suppression
penalty, and tie-breaks near-equal candidates by proximity to the currently
applied control. Only the first step of the winner is applied — receding horizon.

**Bridging simulator to models.** The simulator's physical outputs are mapped onto
the model target names (an arbitrary but reasonable bridge — there is no literal
1:1 correspondence in the training data):

| Model target | Simulator quantity |
| :--- | :--- |
| `return` | `Mr`, the recirculated reject flow (t/h) |
| `first_chamber_filling` | hold-up `H1` as % of `H_cap` |
| `second_chamber_filling` | hold-up `H2` as % of `H_cap` |
| `gran1_blain` | `blaine` |

Controls go the other way: `separator_speed` → `sim.set_rotor_speed()`,
`fresh_feed_setpoint` → the fresh feed rate `Mc` (converted t/h → t/min).

**Time resolution.** The models step at 5 minutes (`STEP_MINUTES`), but the
simulator and the decision loop run at 1 minute (`FINE_DT_MIN`). Rather than wait
five minutes per decision, `_pooled_history` rebuilds the controller's history
every minute from a *rolling* window of raw 1-minute readings, mean-pooled into
5-minute-equivalent blocks. The models always see their usual step size while the
controller replans every minute. Replanning that often is exactly what would cause
chattering between near-equal grid points, which is why `TIE_BREAK_TOLERANCE` and
`MOVE_PENALTY_WEIGHTS` matter here.

**Run structure.** The simulator is warmed up for `WARMUP_MINUTES` at nominal
control, then seeded for `max_input_chunk_length * BLOCK_SIZE` raw minutes so the
controller has a full history. Warm-up and seeding are setpoint-independent and
therefore executed **once**, and every run starts from that identical snapshot —
which is what makes the per-target IAE numbers comparable. Then
`N_CONTROLLED_MINUTES` of closed-loop control follow, one decision per minute.

With `PER_TARGET_RUNS` (`--per-target`, the default here), the script does one run
per `SETPOINTS` entry: the target under test keeps its assigned setpoint while
every other target is pinned to its own value at the end of seeding, so it starts
at exactly zero error. Any control action is then attributable to that one target.

**Tuning knobs** (module-level constants near the top of the file):

| Constant | Meaning |
| :--- | :--- |
| `MODEL_NAMES` | `darts_logs/` directory per target — currently `LinearRegression I6/O3` for `return` and `gran1_blain`, `NeuralForecast_TSMixer I2/O3` for the two fillings. |
| `SETPOINTS` | Target setpoint per variable. |
| `ERROR_METRIC` | `mae` (raw units), `mape` (scale-free), or `itae` (time-weighted, same scale as `mae`). |
| `TARGET_WEIGHTS` | Per-target weight in the combined cost. `gran1_blain` is down-weighted to `0.02` because under `mae` its raw error dwarfs the percentage-scale fillings. |
| `MOVE_PENALTY_WEIGHTS` | Per-control penalty on deviating from the applied control, normalised by that control's bounds range. Competes directly with the error terms. |
| `TIE_BREAK_TOLERANCE` | Fraction of the best cost within which candidates count as tied. |
| `CONTROL_STEPS` | Grid resolution for both controls. |
| `N_CONTROLLED_MINUTES` | Closed-loop length. At ~1–2 s per decision, 500 minutes ≈ 10–15 min wall clock. |
| `NOMINAL_RPM` / `NOMINAL_FEED_TH` | Control held during warm-up and seeding. |
| `ADD_NOISE` / `NOISE_STD` | Gaussian measurement noise on targets, applied only during the controlled phase. |
| `IS_NAIVE` | Swap in the naive predictor, as a controller baseline. |

**Outputs** (written next to the script, suffixed with `_<target>` in a sweep):

- `mpc_state_per_minute*.csv` — one row per simulated minute: phase, all targets and
  controls, each target's setpoint and error, the simulator's raw physical outputs
  under a `sim_` prefix, and the MPC cost plus per-target predicted error for that
  minute.
- `mpc_closed_loop_iae*.csv` — per-target closed-loop IAE, the outer-loop score, and
  deliberately *not* the controller's own metric: it integrates
  `|y(t) - setpoint|` over time with the trapezoid rule. Reported over two windows,
  `full` (including the open-loop seeding phase) and `controlled` (`t >= seed_end_t`),
  each with an `iae_per_minute` normalisation so windows and run lengths are
  comparable. A final `TOTAL_weighted` row combines targets using `TARGET_WEIGHTS`.
- `fig_mpc_closed_loop*.png` — 6-panel figure: the four targets with their setpoints,
  then the two controls as step plots, with a vertical marker where MPC takes over.
  Panels are titled with the exact variable names used in the code and CSVs.

---

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements3.13.5.txt
```

Python 3.13.5. Scripts insert the project root on `sys.path`, so they can be run
either directly (`python model_training/train_models.py`) or as modules
(`python -m mpc2.run_mpc_on_simulator`); the MPC needs the module form.

## License

Released under the [MIT License](LICENSE) — free to use, modify and
redistribute, with attribution and without warranty.
