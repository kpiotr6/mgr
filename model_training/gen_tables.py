import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("result.csv")

# Keep only needed columns in output tables
base_cols = ["InputChunkLength", "Model", "MAE", "RMSE", "MAPE"]

# Round UP to 2 decimals for MAE, RMSE and MAPE
df["MAE"] = np.ceil(df["MAE"] * 100) / 100
df["RMSE"] = np.ceil(df["RMSE"] * 100) / 100
df["MAPE"] = np.ceil(df["MAPE"] * 100) / 100

# Escape underscores in model names for LaTeX
df["Model"] = df["Model"].str.replace("_", r"\_", regex=False)

# Build one .tex per OutputChunkLength, but do not print that column in table
for i, out_len in enumerate(sorted(df["OutputChunkLength"].unique()), start=1):
    sub = df[df["OutputChunkLength"] == out_len][base_cols].copy()
    sub = sub.sort_values(["InputChunkLength", "Model"])
    lookback_values = ", ".join(str(v) for v in sorted(sub["InputChunkLength"].unique()))

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Lookback window (InputChunkLength): {lookback_values}}}"
    )
    lines.append(rf"\label{{tab:lookback_{i}}}")
    lines.append(r"\begin{tabular}{|c|l|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{InputChunkLength} & \textbf{Model} & \textbf{MAE} & \textbf{RMSE} & \textbf{MAPE} \\")
    lines.append(r"\hline")

    for _, row in sub.iterrows():
        lines.append(
            f"{int(row['InputChunkLength'])} & {row['Model']} & {row['MAE']:.2f} & {row['RMSE']:.2f} & {row['MAPE']:.2f} \\\\"
        )
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    # No output length in table content; filename uses index only
    with open(f"table_{i}.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

print("Created files:", [f"table_{i}.tex" for i in range(1, len(df['OutputChunkLength'].unique()) + 1)])