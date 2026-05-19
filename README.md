# Time-Series Utilities

## Cross-Correlation + Granger Causality

Script: `data_functionalities/correlation_granger.py`

This script treats each `session_index` as a separate time series and computes:
- lagged cross-correlation,
- Granger causality (`X -> Y` and `Y -> X`).

### Example

```bash
python data_functionalities/correlation_granger.py \
  --input-file data_preprocessed/data1.csv \
  --x-col separator_speed_setpoint \
  --y-col gran1_blain \
  --max-lag 10 \
  --alpha 0.05 \
  --output-dir outputs
```

### Outputs

- `outputs/cross_corr_granger_summary_<input_stem>.csv`
  - one row per `session_index` and variable pair,
  - contains best cross-correlation lag/value,
  - contains minimum Granger p-value and significance flag in both directions.

- `outputs/cross_corr_lags_<input_stem>.csv`
  - lag-level correlation values for each `session_index` and variable pair.

- `outputs/cross_corr_granger_aggregate_<input_stem>.csv`
  - one row per variable pair (`x_col`, `y_col`) aggregated across all sessions,
  - includes ranking-ready metrics such as `mean_best_corr_abs`, `weighted_mean_best_corr_abs`, `max_best_corr_abs`,
  - includes Granger significance ratios (`x_to_y_significant_ratio`, `y_to_x_significant_ratio`).

### Notes

- If `--x-col` and `--y-col` are omitted, pairs are generated as `INPUT_COLS × TARGET_COLS` from `config.py`.
- Use small `--max-lag` first for faster runtime.

## Darts Training Pipeline

Script: `train_darts_pipeline.py`

Trains Darts models on all CSV files in `data_preprocessed`, treating each
`session_index` as a separate series. Uses `PAST_COLS` as past covariates,
`INPUT_COLS` as future covariates, and `TARGET_COLS` as targets.

### Example

```bash
python train_darts_pipeline.py \
  --input-chunk-length 120 \
  --output-chunk-length 30 \
  --models NeuralForecast_TSMixer,LinearRegression \
  --val-ratio 0.2 \
  --save-dir outputs/models
```

### Notes

- Use `--max-files` for a quick sanity run.
- Models are created via `model_training/model_definitions.py`.

## Linear Regression Forward Selection

Script: `model_training/train_linear_regression_forward_selection.py`

Runs forward selection over `PAST_COLS` and `INPUT_COLS` to pick covariate subsets
for a Darts `LinearRegressionModel` with backward (past) and forward (future) windows.

### Example

```bash
python model_training/train_linear_regression_forward_selection.py \
  --input-chunk-length 60 \
  --output-chunk-length 30 \
  --input-chunk-lengths 60,120 \
  --output-chunk-lengths 15,30 \
  --max-features 10 \
  --min-improvement 1e-4 \
  --shuffle \
  --output-dir outputs
```

### Outputs

- `outputs/linear_regression_forward_selection_steps_<target>_I<in>_O<out>.csv`
  - incremental feature additions with validation RMSE for each target and length pair.
- `outputs/linear_regression_forward_selection_summary.csv`
  - final selected covariates and best validation RMSE for all targets and length pairs.
