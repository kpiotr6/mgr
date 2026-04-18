import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CATEGORY_MAP: Dict[str, str] = {
    "NaiveLastValue": "Baseline",
    "NaiveSeasonal": "Baseline",
    "LinearRegression": "Linear",
    "NLinear": "Linear",
    "DLinear": "Linear",
    "LightGBM": "TreeBased",
    "XGBoost": "TreeBased",
    "XGB": "TreeBased",
    "RandomForest": "TreeBased",
    "CatBoost": "TreeBased",
    "TSMixer": "DeepLearning",
    "NHiTS": "DeepLearning",
    "TFT": "DeepLearning",
    "BlockRNN": "DeepLearning",
    "LSTM": "DeepLearning",
    "RNN": "DeepLearning",
}


def infer_category(model_name: str) -> str:
    if model_name in DEFAULT_CATEGORY_MAP:
        return DEFAULT_CATEGORY_MAP[model_name]

    lower_name = model_name.lower()
    if "naive" in lower_name:
        return "Baseline"
    if "linear" in lower_name:
        return "Linear"
    if any(tag in lower_name for tag in ["lightgbm", "xgb", "forest", "catboost"]):
        return "TreeBased"
    if any(tag in lower_name for tag in ["nhits", "tsmixer", "tft", "rnn", "lstm", "transformer"]):
        return "DeepLearning"
    return "Other"


def validate_columns(df: pd.DataFrame, input_col: str, output_col: str, model_col: str, metric_col: str) -> None:
    required = [input_col, output_col, model_col, metric_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")


def save_grouped_bar_plot(
    frame: pd.DataFrame,
    input_col: str,
    model_col: str,
    metric_col: str,
    title: str,
    output_path: str,
) -> None:
    pivot = frame.pivot_table(index=model_col, columns=input_col, values=metric_col, aggfunc="mean")
    pivot = pivot.sort_index()

    if pivot.empty:
        return

    ax = pivot.plot(kind="bar", figsize=(12, 7), width=0.85)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_col)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title=input_col, loc="best")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_plots(
    csv_path: str,
    metric_col: str,
    out_dir: str,
    input_col: str = "InputChunkLength",
    output_col: str = "OutputChunkLength",
    model_col: str = "Model",
) -> None:
    df = pd.read_csv(csv_path)
    validate_columns(df, input_col, output_col, model_col, metric_col)

    df["ModelCategory"] = df[model_col].apply(infer_category)

    output_lengths = sorted(df[output_col].dropna().unique())
    for output_length in output_lengths:
        subset = df[df[output_col] == output_length]
        if subset.empty:
            continue

        output_plot = os.path.join(out_dir, f"bar_{metric_col}_O{int(output_length)}.png")
        title = f"{metric_col} by Model (OutputChunkLength={int(output_length)})"
        save_grouped_bar_plot(
            subset,
            input_col=input_col,
            model_col=model_col,
            metric_col=metric_col,
            title=title,
            output_path=output_plot,
        )

        for category in sorted(subset["ModelCategory"].dropna().unique()):
            category_subset = subset[subset["ModelCategory"] == category]
            if category_subset.empty:
                continue

            category_slug = category.lower().replace(" ", "_")
            category_plot = os.path.join(
                out_dir,
                f"bar_{metric_col}_{category_slug}_O{int(output_length)}.png",
            )
            category_title = (
                f"{metric_col} by Model ({category}, OutputChunkLength={int(output_length)})"
            )
            save_grouped_bar_plot(
                category_subset,
                input_col=input_col,
                model_col=model_col,
                metric_col=metric_col,
                title=category_title,
                output_path=category_plot,
            )

    print(f"Bar plots saved to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate bar plots per output chunk length and per model category + output chunk length."
        )
    )
    parser.add_argument(
        "--csv",
        default="result.csv",
        help="Path to input metrics CSV file (default: result.csv)",
    )
    parser.add_argument(
        "--metric",
        default="MAE",
        help="Metric column to plot on y-axis (default: MAE)",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/outputchunk_bars",
        help="Directory for generated plots (default: plots/outputchunk_bars)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_plots(csv_path=args.csv, metric_col=args.metric, out_dir=args.out_dir)
