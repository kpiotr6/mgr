import os
import glob
import pandas as pd
import numpy as np

def check_model_bias(outputs_dir="outputs"):
    print(f"--- Model Bias Analysis (Overshoot vs Undershoot) ---")
    print(f"{'Model':<20} | {'Config':<10} | {'Mean Error':<12} | {'Overshoot %':<12} | {'Undershoot %':<12}")
    print("-" * 75)

    results = []

    # Find all prediction files
    pred_files = glob.glob(os.path.join(outputs_dir, "pred_*.csv"))

    for pred_file in sorted(pred_files):
        filename = os.path.basename(pred_file)
        # Parse filename, e.g., pred_DLinear_I30_O30.csv
        # Remove 'pred_' and '.csv'
        base_name = filename[5:-4]  # DLinear_I30_O30

        # Split into model name and config. Assuming config is always I..._O...
        parts = base_name.split('_')

        # Find where the 'I' part starts
        config_idx = -1
        for i, part in enumerate(parts):
            if part.startswith('I') and 'O' in parts[min(i+1, len(parts)-1)]:
                config_idx = i
                break

        if config_idx != -1:
            model_name = "_".join(parts[:config_idx])
            config = "_".join(parts[config_idx:])
        else:
            # Fallback parsing
            model_name = base_name
            config = "unknown"

        true_file = os.path.join(outputs_dir, f"true_values_{config}.csv")

        if not os.path.exists(true_file):
            print(f"Warning: True values file {true_file} not found for {filename}")
            continue

        try:
            pred_df = pd.read_csv(pred_file)
            true_df = pd.read_csv(true_file)

            # Use only numeric columns to avoid index/date columns if they exist
            pred_num = pred_df.select_dtypes(include=[np.number]).values.flatten()
            true_num = true_df.select_dtypes(include=[np.number]).values.flatten()

            if len(pred_num) != len(true_num):
                # If there's a size mismatch, try dropping NaNs just in case
                pred_num = pred_num[~np.isnan(pred_num)]
                true_num = true_num[~np.isnan(true_num)]

                if len(pred_num) != len(true_num):
                    print(f"{model_name:<20} | {config:<10} | Size mismatch error")
                    continue

            errors = pred_num - true_num

            mean_error = np.mean(errors)
            overshoot_pct = np.sum(errors > 0) / len(errors) * 100
            undershoot_pct = np.sum(errors < 0) / len(errors) * 100

            print(f"{model_name:<20} | {config:<10} | {mean_error:>12.4f} | {overshoot_pct:>11.2f}% | {undershoot_pct:>11.2f}%")

            results.append({
                "Model": model_name,
                "Config": config,
                "Mean_Error": mean_error,
                "Overshoot_Pct": overshoot_pct,
                "Undershoot_Pct": undershoot_pct
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save results
    if results:
        res_df = pd.DataFrame(results)
        out_file = os.path.join(outputs_dir, "model_bias_summary.csv")
        res_df.to_csv(out_file, index=False)
        print("-" * 75)
        print(f"Summary saved to {out_file}")

if __name__ == "__main__":
    check_model_bias()
