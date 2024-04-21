import pandas as pd

def filter_empty_columns(filepath):
    df = pd.read_csv(filepath)

    # Drop columns with any empty cells
    df_filtered = df.dropna(axis=1)

    df_filtered.to_csv("Mata-P3D-Manual-columns.csv", index=False)

    return df_filtered

csv_filepath = "Mata-P3D-Manual.csv"
filtered_columns_df = filter_empty_columns(csv_filepath)



