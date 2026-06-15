## Usage

Run the script from your terminal:

```bash
python preprocess.py --input-dir raw_data --output-dir processed_data --ma-window 5 --check-gran-empty
```

### Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input-dir` | String | `data` | The directory path containing the raw `.csv` files to process. |
| `--output-dir` | String | `data_preprocessed` | The directory path where the processed `.csv` files will be saved. |
| `--ma-window` | Integer | `0` | The rolling window length for computing moving averages. `0` disables this feature. |
|
| `--check-gran-empty` | Flag | `False` | Truncates each session at the first instance where `gran1_blain` is `0` or `NaN`. |

## Model Training

Run the model training script from your terminal:

```bash
python model_training/train_models.py --runs all --use-detrend --ma-window 5
```

### Command-Line Arguments for Model Training

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--use-detrend` | Flag | `False` | Apply linear detrending before training and reverse it after prediction. |
| `--use-log-transform` | Flag | `False` | Apply log transform before training and reverse it after prediction. |
| `--shuffle` | Flag | `False` | Shuffle the data blocks before train/val/test split. |
| `--runs` | String (choices) | `all` | Which training runs to execute. Choices: `all`, `all_targets`, `per_target`, `simple`. |
| `--models` | String | `None` | Comma-separated list of model names to train (e.g. `LinearRegression,NeuralForecast_Nhits`). |
| `--exclude-models` | String | `None` | Comma-separated list of model names to exclude. |
| `--model-groups` | String (choices) | `None` | Train only selected model groups. Choices: `naive`, `checkpoint`, `future_covariates`, `future_only`, `other`. |
| `--ma-window` | Integer | `0` | If > 0, calculates moving average column names. Target MAs are added to PAST_COLS to prevent data leakage. |