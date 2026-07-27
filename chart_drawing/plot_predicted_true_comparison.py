import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import glob
from collections import defaultdict

# 1. Define Directories & Constants
INPUT_DIR = 'outputs_ready'
BASE_OUTPUT_DIR = 'chart_drawing/predicted_true_comparison'
MAX_WINDOWS = 10  # Maximum number of forecast windows per plot
SECONDS_PER_STEP = 10  # Each sequence step represents 10 seconds

# 2. Variables to plot
TARGET_VARS = ['return', 'first_chamber_filling', 'second_chamber_filling']

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
    # 3. Find all prediction files matching the new target-specific pattern
    pred_files = glob.glob(os.path.join(INPUT_DIR, 'pred_target_*_*_I*_O*.csv'))

    if not pred_files:
        print(f"No prediction files found in {INPUT_DIR}/ matching the pattern.")
        return

    # Group files by (Input, Output, Model) -> {Target: FilePath}
    configs = defaultdict(dict)

    for p_file in pred_files:
        filename = os.path.basename(p_file)

        # Identify which target this file belongs to
        matched_target = None
        for var in TARGET_VARS:
            if filename.startswith(f"pred_target_{var}_"):
                matched_target = var
                break

        if not matched_target:
            print(f"Skipping {filename}: Target not found in TARGET_VARS.")
            continue

        # Parse the rest of the filename (e.g., model_name_I60_O10.csv)
        rest_of_filename = filename[len(f"pred_target_{matched_target}_"):]
        match = re.match(r"(.*)_I(\d+)_O(\d+)\.csv", rest_of_filename)

        if match:
            model_name, input_len, output_len = match.groups()
            configs[(int(input_len), int(output_len), model_name)][matched_target] = p_file
        else:
            print(f"Skipping {filename}: Does not match naming convention.")

    # 4. Process each Configuration Group (Input, Output, Model)
    for (input_len, output_len, model_name), targets_dict in configs.items():
        print(f"\n{'='*50}\nProcessing Model: {model_name} | Input={input_len}, Output={output_len}\n{'='*50}")

        window_minutes = (output_len * SECONDS_PER_STEP) / 60.0

        dfs_pred = {}
        dfs_true = {}

        # 5. Load DataFrames for all targets in this config
        for var, p_file in targets_dict.items():
            df_p = pd.read_csv(p_file)
            dfs_pred[var] = clean_sequence_index(df_p)

            true_filename = f'true_values_target_{var}_I{input_len}_O{output_len}.csv'
            true_file_path = os.path.join(INPUT_DIR, true_filename)

            if os.path.exists(true_file_path):
                df_t = pd.read_csv(true_file_path)
                dfs_true[var] = clean_sequence_index(df_t)
            else:
                print(f"  -> Warning: Could not find matching true values file {true_filename}.")

        # Find all unique block indices across the loaded target predictions
        all_blocks = set()
        for df in dfs_pred.values():
            if 'block_idx' in df.columns:
                all_blocks.update(df['block_idx'].unique())

        if not all_blocks:
            print(f"Warning: 'block_idx' missing. Skipping config...")
            continue

        # 6. Group and plot by block_idx
        for block_id in sorted(all_blocks):
            io_folder_name = f"I{input_len}_O{output_len}"
            block_output_dir = os.path.join(BASE_OUTPUT_DIR, io_folder_name, f'block_{block_id}')
            os.makedirs(block_output_dir, exist_ok=True)

            ref_var = list(targets_dict.keys())[0]
            df_ref = dfs_pred[ref_var]
            df_block_ref = df_ref[df_ref['block_idx'] == block_id]

            if df_block_ref.empty:
                continue

            # Determine the starting sequence_index for each forecast origin
            origin_to_start_seq = df_block_ref.groupby('fcst_origin')['sequence_index'].min().to_dict()

            # USE FIRST SEQUENCE INDEX AS THE STARTING ALIGNMENT POINT
            start_seq_idx = df_block_ref['sequence_index'].min()

            origins = sorted(origin_to_start_seq.keys())
            if not origins:
                continue

            # Select origins where the starting sequence_index aligns cleanly with the very first sequence_index
            selected_origins = [
                o for o in origins
                if (origin_to_start_seq[o] - start_seq_idx) % output_len == 0
            ]

            # 7. Split the origins into chunks of max windows
            origin_chunks = list(chunk_list(selected_origins, MAX_WINDOWS))
            available_vars = list(targets_dict.keys())
            plot_vars = [v for v in TARGET_VARS if v in available_vars]

            for part_idx, chunk in enumerate(origin_chunks):
                print(f"  -> Generating plot for Block {block_id} (Part {part_idx + 1}/{len(origin_chunks)})...")

                fig, axes = plt.subplots(len(plot_vars), 1, figsize=(18, 4 * len(plot_vars)), sharex=True)
                if len(plot_vars) == 1:
                    axes = [axes]

                # 8. Plot each variable
                for i, var in enumerate(plot_vars):
                    ax = axes[i]

                    # True Values
                    if var in dfs_true:
                        df_t = dfs_true[var]
                        df_t_filt = df_t[(df_t['block_idx'] == block_id) & (df_t['fcst_origin'].isin(chunk))].copy()
                        df_t_filt = df_t_filt.sort_values('sequence_index')

                        y_col_t = var if var in df_t_filt.columns else df_t_filt.columns[-1]

                        ax.plot(df_t_filt['sequence_index'], df_t_filt[y_col_t],
                                label='True Values', linestyle='-', color='tab:blue',
                                linewidth=1.2, marker='o', markersize=4)

                    # Predicted Values
                    df_p = dfs_pred[var]
                    df_p_filt = df_p[(df_p['block_idx'] == block_id) & (df_p['fcst_origin'].isin(chunk))].copy()
                    df_p_filt = df_p_filt.sort_values('sequence_index')

                    y_col_p = var if var in df_p_filt.columns else df_p_filt.columns[-1]

                    ax.plot(df_p_filt['sequence_index'], df_p_filt[y_col_p],
                            label=f'Predicted ({model_name})', linestyle=':', color='tab:orange',
                            linewidth=1.5, marker='o', markersize=4)

                    # Vertical black lines at the start of each forecast window
                    for idx, origin_point in enumerate(chunk):
                        label = f'Forecast Window ({output_len} steps / {window_minutes:.1f} mins)' if idx == 0 else ""
                        window_start_seq = origin_to_start_seq[origin_point]
                        ax.axvline(x=window_start_seq, color='black', linestyle='-', linewidth=1.2, alpha=0.7, label=label)

                    window_minutes_input = (input_len * SECONDS_PER_STEP) / 60.0

                    title_str = f'{var.replace("_", " ").title()} - {model_name} (Block {block_id}, Part {part_idx + 1}) | Lookback Window: ({input_len} steps / {window_minutes_input:.1f} mins)'
                    ax.set_title(title_str, fontsize=14, fontweight='bold')
                    ax.set_ylabel(var)
                    ax.grid(True, linestyle='--', alpha=0.5)

                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))

                    # External legend
                    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)

                axes[-1].set_xlabel('Sequence Index', fontsize=12)
                plt.tight_layout()

                # 9. Save the chart
                output_path = os.path.join(block_output_dir, f'{model_name}_part_{part_idx + 1}.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()

if __name__ == "__main__":
    main()