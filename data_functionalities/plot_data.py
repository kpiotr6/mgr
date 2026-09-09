import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import argparse

features_to_plot = [
    "return",
    "separator_speed",
    "gran1_blain",
]

# Add features you want highlighted in bold here
special_features = [
    "return"
]

def save_plot(df, y_cols, title, plot_path, special_cols, rescale=False):
    plot_df = df[y_cols].copy()

    # Only perform 0 to 1 scaling if the flag is provided
    if rescale:
        for col in y_cols:
            col_min = plot_df[col].min()
            col_max = plot_df[col].max()
            col_range = col_max - col_min
            if col_range == 0:
                plot_df[col] = 0.0
            else:
                plot_df[col] = (plot_df[col] - col_min) / col_range

    plt.figure(figsize=(14, 8))

    for col in y_cols:
        # Check if the feature is in the special list to adjust line thickness and opacity
        if col in special_cols:
            plt.plot(df.index, plot_df[col], label=f"{col} (Bold)", alpha=1.0, linewidth=6.0)
        else:
            plt.plot(df.index, plot_df[col], label=col, alpha=1, linewidth=3.0)

    plt.xlabel('Index')
    # Update y-label dynamically based on scaling
    plt.ylabel('Scaled Value (0 to 1)' if rescale else 'Value')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot: {plot_path}")

def main():
    # Set up argument parsing for the --rescale flag
    parser = argparse.ArgumentParser(description="Plot time series features from CSV files.")
    parser.add_argument(
        "--rescale",
        action="store_true",
        help="If provided, rescales the features from 0 to 1."
    )
    args = parser.parse_args()

    data_dir = "data_preprocessed"
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    # Create an output directory for the plots
    os.makedirs("plots", exist_ok=True)

    for file in files:
        df = pd.read_csv(file)

        # Also drop any Unnamed columns that might have slipped in
        y_cols = [col for col in features_to_plot if col in df.columns]

        if 'session_index' in df.columns:
            session_change_group = (df['session_index'] != df['session_index'].shift()).cumsum()

            for segment_id, session_df in df.groupby(session_change_group, sort=False):
                session_value = session_df['session_index'].iloc[0]
                base_name = os.path.basename(file).replace('.csv', '')
                plot_name = f"{base_name}_session_{session_value}_segment_{segment_id}.png"
                plot_path = os.path.join("plots", plot_name)
                plot_title = f"Values from {os.path.basename(file)} | session_index={session_value} | segment={segment_id}"

                save_plot(session_df, y_cols, plot_title, plot_path, special_features, rescale=args.rescale)
        else:
            plot_path = os.path.join("plots", os.path.basename(file).replace('.csv', '.png'))
            save_plot(df, y_cols, f'Values from {os.path.basename(file)}', plot_path, special_features, rescale=args.rescale)
            print(f"Warning: 'session_index' not found in {file}, created a single plot.")

if __name__ == "__main__":
    main()