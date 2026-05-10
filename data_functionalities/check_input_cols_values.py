import pandas as pd
import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import INPUT_COLS

def get_unique_values(data_dir="data"):
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not all_files:
        print(f"No CSV files found in {data_dir}")
        return

    unique_values = {col: set() for col in INPUT_COLS}

    for file in all_files:
        df = pd.read_csv(file)

        for col in INPUT_COLS:
            if col in df.columns:
                unique_values[col].update(df[col].dropna().unique())

    for col in INPUT_COLS:
        print(f"--- Unique values for {col} ---")
        values = sorted(list(unique_values[col]))
        print(values)
        print(f"Total unique values: {len(values)}\n")

if __name__ == "__main__":
    print("Checking raw data:")
    get_unique_values("data")
    print("=========================================")
    print("Checking preprocessed data:")
    get_unique_values("data_preprocessed")
