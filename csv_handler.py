import glob
import os
import pandas as pd

def combine_csv_files(folder_path, output_file):
    # Find and combine all CSV files in the folder
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    combined_df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    # Export
    combined_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    combine_csv_files(folder_path="card_data/sets/", output_file="./card_data/compiled_data.csv")
