import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def save_plot(df, y_cols, title, plot_path):
    scaled_df = df[y_cols].copy()
    for col in y_cols:
        col_min = scaled_df[col].min()
        col_max = scaled_df[col].max()
        col_range = col_max - col_min
        if col_range == 0:
            scaled_df[col] = 0.0
        else:
            scaled_df[col] = (scaled_df[col] - col_min) / col_range

    plt.figure(figsize=(14, 8))

    for col in y_cols:
        plt.plot(df.index, scaled_df[col], label=col, alpha=0.8)

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot: {plot_path}")

def main():
    data_dir = "data_preprocessed"
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    # Create an output directory for the plots
    os.makedirs("plots", exist_ok=True)

    for file in files:
        df = pd.read_csv(file)

        cols_to_drop = ['is_at_edge', 'time', 'sequence_index']
        # Also drop any Unnamed columns that might have slipped in
        y_cols = [col for col in df.columns if col not in cols_to_drop and not col.startswith('Unnamed')]

        if 'session_index' in df.columns:
            session_change_group = (df['session_index'] != df['session_index'].shift()).cumsum()

            for segment_id, session_df in df.groupby(session_change_group, sort=False):
                session_value = session_df['session_index'].iloc[0]
                base_name = os.path.basename(file).replace('.csv', '')
                plot_name = f"{base_name}_session_{session_value}_segment_{segment_id}.png"
                plot_path = os.path.join("plots", plot_name)
                plot_title = f"Values from {os.path.basename(file)} | session_index={session_value} | segment={segment_id}"
                save_plot(session_df, y_cols, plot_title, plot_path)
        else:
            plot_path = os.path.join("plots", os.path.basename(file).replace('.csv', '.png'))
            save_plot(df, y_cols, f'Values from {os.path.basename(file)}', plot_path)
            print(f"Warning: 'session_index' not found in {file}, created a single plot.")

if __name__ == "__main__":
    main()
