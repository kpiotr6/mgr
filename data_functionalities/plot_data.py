import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def main():
    data_dir = "data_preprocessed"
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    # Create an output directory for the plots
    os.makedirs("plots", exist_ok=True)

    for file in files:
        df = pd.read_csv(file)

        cols_to_drop = ['is_at_edge', 'time']
        # Also drop any Unnamed columns that might have slipped in
        y_cols = [col for col in df.columns if col not in cols_to_drop and not col.startswith('Unnamed')]

        plt.figure(figsize=(14, 8))

        for col in y_cols:
            plt.plot(df.index, df[col], label=col, alpha=0.8)

        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'Values from {os.path.basename(file)}')

        # Move legend outside of the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plot_path = os.path.join("plots", os.path.basename(file).replace('.csv', '.png'))
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")

if __name__ == "__main__":
    main()
