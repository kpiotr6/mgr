import pandas as pd
import glob

def find_empty_columns(file_pattern="data/*.csv"):
    """
    Finds and prints columns with empty (null) values in CSV files matching the given pattern.
    """
    files = glob.glob(file_pattern)

    if not files:
        print(f"No files found matching the pattern: {file_pattern}")
        return

    all_empty_cols = set()

    for file in files:
        print(f"\nChecking file: {file}")
        try:
            df = pd.read_csv(file)
            empty_cols = df.columns[df.isnull().any()].tolist()

            if empty_cols:
                all_empty_cols.update(empty_cols)
                print("Columns with empty values:")
                for col in empty_cols:
                    num_empty = df[col].isnull().sum()
                    print(f"  - {col}: {num_empty} missing values")
            else:
                print("No empty values found in any column.")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    print("\n" + "="*40)
    if all_empty_cols:
        print("SUMMARY: All columns with missing values across all files:")
        for col in sorted(all_empty_cols):
            print(f"  - {col}")
    else:
        print("SUMMARY: No empty values found across all files.")
    print("="*40 + "\n")

if __name__ == "__main__":
    find_empty_columns()