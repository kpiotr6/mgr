# MPC (discrete grid search)

Files:
- `mpc/simulator_headless.py`: `CementMillSim` headless simulator.
- `mpc/mpc_controller.py`: `DartsPredictor` + `DiscreteGridMPC`.
- `mpc/run_closed_loop.py`: runs 60-step receding-horizon closed-loop and prints MAE/MSE/MAPE.
- `mpc/fit_scalers.py`: creates `darts_logs/<model_name>/scalers.pkl` for a checkpoint folder.

## Expected columns
Controls (optimized by MPC):
- `clinker_1_feedrate` (0..120)
- `separator_speed` (280..800)

Targets (multivariate model outputs):
- `return`
- `first_chamber_filling`
- `second_chamber_filling`
- `gran1_blain`

## 1) Ensure scalers exist
MPC requires `darts_logs/<model_name>/scalers.pkl`.

```bash
/home/kpiotr6/Documents/stuida/praca_magisterska/proj/.venv/bin/python -m mpc.fit_scalers \
  --model-name <YOUR_MODEL_NAME> \
  --input-cols clinker_1_feedrate separator_speed \
  --target-cols return first_chamber_filling second_chamber_filling gran1_blain
```

## 2) Run closed-loop (60 steps)
If you don’t pass setpoints, it regulates to the warmup steady-state.

```bash
/home/kpiotr6/Documents/stuida/praca_magisterska/proj/.venv/bin/python -m mpc.run_closed_loop \
  --model-name <YOUR_MODEL_NAME> \
  --model-class NeuralForecastModel \
  --forward-horizon 30 \
  --grid-n 5
```

Custom setpoints (JSON string or a JSON file path):

```bash
/home/kpiotr6/Documents/stuida/praca_magisterska/proj/.venv/bin/python -m mpc.run_closed_loop \
  --model-name <YOUR_MODEL_NAME> \
  --setpoints '{"return":120,"first_chamber_filling":85,"second_chamber_filling":70,"gran1_blain":3600}'
```
