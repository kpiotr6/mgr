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

# Define a color palette for the models so True is always black/dark blue
# Matplotlib's default tab colors, excluding blue (reserved for True Values)
MODEL_COLORS = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

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

        # Calculate the time equivalents
        window_minutes_out = (output_len * SECONDS_PER_STEP) / 60.0
        window_minutes_in = (input_len * SECONDS_PER_STEP) / 60.0

        # Load True Values
        true_filename = f'true_values_all_targets_I{input_len}_O{output_len}.csv'
        true_file_path = os.path.join(INPUT_DIR, true_filename)

        if not os.path.exists(true_file_path):
            print(f"  -> Error: Could not find matching true values file {true_filename}. Skipping config.")
            continue

        df_true = clean_sequence_index(pd.read_csv(true_file_path))

        # Load all models for this configuration into memory at once
        model_data = {}
        for p_file, model_name in models:
            df_pred = clean_sequence_index(pd.read_csv(p_file))
            if 'block_idx' in df_pred.columns:
                model_data[model_name] = df_pred
            else:
                print(f"Warning: 'block_idx' not found in {os.path.basename(p_file)}. Skipping model {model_name}...")

        if not model_data:
            print("No valid models to plot for this configuration.")
            continue

        # 5. Group by block_idx using the True Values as the source of truth for blocks
        for block_id, df_true_block in df_true.groupby('block_idx'):

            # Build nested folder structure
            io_folder_name = f"I{input_len}_O{output_len}"
            block_output_dir = os.path.join(BASE_OUTPUT_DIR, io_folder_name, f'block_{block_id}')
            os.makedirs(block_output_dir, exist_ok=True)

            origins = sorted(df_true_block['fcst_origin'].unique())
            if not origins:
                continue

            start_origin = origins[0]
            selected_origins = [o for o in origins if (o - start_origin) % output_len == 0]

            # 6. Split the origins into chunks
            origin_chunks = list(chunk_list(selected_origins, MAX_WINDOWS))

            for part_idx, chunk in enumerate(origin_chunks):
                print(f"  -> Generating combined plot for Block {block_id} (Part {part_idx + 1}/{len(origin_chunks)})...")

                df_true_filtered = df_true_block[df_true_block['fcst_origin'].isin(chunk)].copy()
                df_true_filtered = df_true_filtered.sort_values('sequence_index')

                fig, axes = plt.subplots(len(TARGET_VARS), 1, figsize=(18, 4 * len(TARGET_VARS)), sharex=True)
                if len(TARGET_VARS) == 1:
                    axes = [axes]

                # 7. Plot each variable
                for i, var in enumerate(TARGET_VARS):
                    ax = axes[i]

                    # Plot True Values first
                    if var in df_true_filtered.columns:
                        ax.plot(df_true_filtered['sequence_index'], df_true_filtered[var],
                                label='True Values', linestyle='-', color='tab:blue',
                                linewidth=2.0, marker='o', markersize=5, zorder=3)

                    # Plot all models for this variable
                    for model_idx, (model_name, df_pred) in enumerate(model_data.items()):
                        color = MODEL_COLORS[model_idx % len(MODEL_COLORS)]

                        df_pred_filtered = df_pred[(df_pred['block_idx'] == block_id) &
                                                   (df_pred['fcst_origin'].isin(chunk))].copy()
                        df_pred_filtered = df_pred_filtered.sort_values('sequence_index')

                        if var in df_pred_filtered.columns and not df_pred_filtered.empty:
                            ax.plot(df_pred_filtered['sequence_index'], df_pred_filtered[var],
                                    label=f'Predicted ({model_name})', linestyle=':', color=color,
                                    linewidth=1.5, marker='x', markersize=4, alpha=0.8, zorder=2)

                    # Vertical black lines for Forecast Origins
                    for idx, origin_point in enumerate(chunk):
                        label = f'Forecast Windows' if idx == 0 else ""
                        ax.axvline(x=origin_point, color='black', linestyle='-', linewidth=1.2, alpha=0.7, label=label, zorder=1)

                    # Fixed Title to show both Input (Lookback) and Output (Forecast)
                    title_str = (f'{var.replace("_", " ").title()} - All Models (Block {block_id}, Part {part_idx + 1}) | '
                                 f'Lookback: {input_len} steps ({window_minutes_in:.1f}m) | '
                                 f'Forecast: {output_len} steps ({window_minutes_out:.1f}m)')
                    ax.set_title(title_str, fontsize=14, fontweight='bold')
                    ax.set_ylabel(var)
                    ax.grid(True, linestyle='--', alpha=0.5)

                    # Deduplicate legend items based on label
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)

                axes[-1].set_xlabel('Sequence Index', fontsize=12)
                plt.tight_layout()

                # 8. Save the combined chart
                output_path = os.path.join(block_output_dir, f'all_models_part_{part_idx + 1}.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()

if __name__ == "__main__":
    main()