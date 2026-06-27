import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import glob
from collections import defaultdict

# 1. Define Directories & Constants
INPUT_DIR = 'outputs_ready'
BASE_OUTPUT_DIR = 'chart_drawing/predicted_true_comparison'
MAX_WINDOWS = 5  # Maximum number of forecast windows per plot
SECONDS_PER_STEP = 10  # Each sequence step represents 10 seconds

# 2. Variables to plot
TARGET_VARS = ['return', 'first_chamber_filling', 'second_chamber_filling', 'gran1_blain']

def clean_sequence_index(df):
    """
    Converts the malformed datetime string back to the original integer sequence index.
    """
    if 'sequence_index' in df.columns and df['sequence_index'].dtype == 'object':
        df['sequence_index'] = pd.to_datetime(df['sequence_index']).astype('int64')
    return df

def chunk_list(lst, n):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    # 3. Find all prediction files
    # Pattern looks for: pred_all_targets_<model_name>_I<input>_O<output>.csv
    pred_files = glob.glob(os.path.join(INPUT_DIR, 'pred_all_targets_*_I*_O*.csv'))

    if not pred_files:
        print(f"No prediction files found in {INPUT_DIR}/ matching the I/O pattern.")
        return

    # Group files by (Input, Output) length configuration
    files_by_io = defaultdict(list)
    filename_pattern = re.compile(r"pred_all_targets_(.*)_I(\d+)_O(\d+)\.csv")

    for p_file in pred_files:
        filename = os.path.basename(p_file)
        match = filename_pattern.match(filename)
        if match:
            model_name, input_len, output_len = match.groups()
            files_by_io[(int(input_len), int(output_len))].append((p_file, model_name))
        else:
            print(f"Skipping {filename}: Does not match naming convention.")

    # 4. Process each I/O configuration group
    for (input_len, output_len), models in files_by_io.items():
        print(f"\n{'='*50}\nProcessing Configuration: Input={input_len}, Output={output_len}\n{'='*50}")

        # Calculate the time equivalent for the output window
        window_minutes = (output_len * SECONDS_PER_STEP) / 60.0

        # Load True Values for this specific I/O configuration
        true_filename = f'true_values_all_targets_I{input_len}_O{output_len}.csv'
        true_file_path = os.path.join(INPUT_DIR, true_filename)

        if not os.path.exists(true_file_path):
            print(f"  -> Error: Could not find matching true values file {true_filename}. Skipping this config.")
            continue

        df_true = pd.read_csv(true_file_path)
        df_true = clean_sequence_index(df_true)

        # 5. Process each model inside this configuration
        for p_file, model_name in models:
            print(f"\nProcessing model: {model_name} (I{input_len}/O{output_len})...")

            df_pred = pd.read_csv(p_file)
            df_pred = clean_sequence_index(df_pred)

            if 'block_idx' not in df_pred.columns:
                print(f"Warning: 'block_idx' not found in {os.path.basename(p_file)}. Skipping...")
                continue

            # 6. Group by block_idx
            for block_id, df_block in df_pred.groupby('block_idx'):

                # Build nested folder structure
                io_folder_name = f"I{input_len}_O{output_len}"
                block_output_dir = os.path.join(BASE_OUTPUT_DIR, io_folder_name, f'block_{block_id}')
                os.makedirs(block_output_dir, exist_ok=True)

                origins = sorted(df_block['fcst_origin'].unique())
                if not origins:
                    continue

                start_origin = origins[0]
                selected_origins = [o for o in origins if (o - start_origin) % output_len == 0]

                # 7. Split the origins into chunks of max windows
                origin_chunks = list(chunk_list(selected_origins, MAX_WINDOWS))

                for part_idx, chunk in enumerate(origin_chunks):
                    print(f"  -> Generating {model_name} plot for Block {block_id} (Part {part_idx + 1}/{len(origin_chunks)})...")

                    df_pred_filtered = df_block[df_block['fcst_origin'].isin(chunk)].copy()
                    df_pred_filtered = df_pred_filtered.sort_values('sequence_index')

                    df_true_filtered = df_true[(df_true['block_idx'] == block_id) & (df_true['fcst_origin'].isin(chunk))].copy()
                    df_true_filtered = df_true_filtered.sort_values('sequence_index')

                    # Slightly increased figure width to accommodate the external legend
                    fig, axes = plt.subplots(len(TARGET_VARS), 1, figsize=(18, 4 * len(TARGET_VARS)), sharex=True)
                    if len(TARGET_VARS) == 1:
                        axes = [axes]

                    # 8. Plot each variable
                    for i, var in enumerate(TARGET_VARS):
                        ax = axes[i]

                        # Plot True Values WITH DOTS
                        if var in df_true_filtered.columns:
                            ax.plot(df_true_filtered['sequence_index'], df_true_filtered[var],
                                    label='True Values', linestyle='-', color='tab:blue',
                                    linewidth=1.2, marker='o', markersize=4)

                        # Plot Predicted Values WITH DOTS
                        if var in df_pred_filtered.columns:
                            ax.plot(df_pred_filtered['sequence_index'], df_pred_filtered[var],
                                    label=f'Predicted ({model_name})', linestyle=':', color='tab:orange',
                                    linewidth=1.5, marker='o', markersize=4)

                        # Vertical black lines
                        for idx, origin_point in enumerate(chunk):
                            label = f'Forecast Window ({output_len} steps / {window_minutes:.1f} mins)' if idx == 0 else ""
                            ax.axvline(x=origin_point, color='black', linestyle='-', linewidth=1.2, alpha=0.7, label=label)

                        window_minutes_input = (input_len * SECONDS_PER_STEP) / 60.0

                        title_str = f'{var.replace("_", " ").title()} - {model_name} (Block {block_id}, Part {part_idx + 1}) | Lookback Window: ({input_len} steps / {window_minutes_input:.1f} mins)'
                        ax.set_title(title_str, fontsize=14, fontweight='bold')
                        ax.set_ylabel(var)
                        ax.grid(True, linestyle='--', alpha=0.5)

                        handles, labels = ax.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))

                        # NEW: Push legend completely outside the plot area to the right
                        ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)

                    axes[-1].set_xlabel('Sequence Index', fontsize=12)
                    plt.tight_layout()

                    # 9. Save the chart with bbox_inches='tight' so the external legend isn't clipped
                    output_path = os.path.join(block_output_dir, f'{model_name}_part_{part_idx + 1}.png')
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close()

if __name__ == "__main__":
    main()