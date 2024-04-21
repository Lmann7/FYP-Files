import pandas as pd

def filter_empty_cells(filepath):
    df = pd.read_csv(filepath)

    # Filter out rows with any empty cells
    df_filtered = df.dropna()

    df_filtered.to_csv("Mata-P3D-Manual-rows.csv", index=False)

    return df_filtered

csv_filepath = "Mata-P3D-Manual.csv"
filtered_df = filter_empty_cells(csv_filepath)
