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
