import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def main():
    # Load data
    csv_path = 'outputs/evaluation_metrics.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Metrics to plot
    metrics = ['MAE', 'RMSE', 'MAPE', 'MAE_gran1_blain', 'RMSE_gran1_blain', 'MAPE_gran1_blain']

    # Ensure output directory exists
    out_dir = 'plots/metrics_bars'
    os.makedirs(out_dir, exist_ok=True)

    # Create a combined column for input/output chunk lengths for clearer grouping
    df['Chunk (In/Out)'] = df['InputChunkLength'].astype(str) + '/' + df['OutputChunkLength'].astype(str)

    chunks = df['Chunk (In/Out)'].unique()
    models = df['Model'].unique()

    for metric in metrics:
        if metric not in df.columns:
            continue

        plt.figure(figsize=(14, 7))

        # Calculate bar positions
        x = np.arange(len(chunks))
        width = 0.8 / len(models)  # Dynamically adjust width based on number of models

        # Plot bars for each model
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]

            # Map values to corresponding chunk order
            values = []
            for chunk in chunks:
                val = model_data[model_data['Chunk (In/Out)'] == chunk][metric]
                if not val.empty:
                    values.append(val.values[0])
                else:
                    values.append(0)

            plt.bar(x + i * width - 0.4 + width / 2, values, width, label=model)

        ax = plt.gca()
        # Using logarithmic scale because some models (e.g., NeuralForecast_Nhits)
        # have errors several orders of magnitude larger than others.
        ax.set_yscale('log')

        plt.title(f'{metric} by Input/Output Chunk Length and Model (Log Scale)')
        plt.ylabel(f'{metric} (log scale)')
        plt.xlabel('Input / Output Chunk Length')

        # Set x-ticks properly
        plt.xticks(x, chunks)

        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        out_file = os.path.join(out_dir, f'{metric}_bar_plot.png')
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved plot: {out_file}")

if __name__ == '__main__':
    main()
