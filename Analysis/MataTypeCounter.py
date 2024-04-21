import pandas as pd

# Read the CSV data
df = pd.read_csv('Mata-P3D-2.6-S2-imputed.csv')

# Count occurrences in the entire dataset (unchanged)
type_counts_whole = df['Type_x'].value_counts()

# Print the whole dataset counts (unchanged)
print("\nCounts in whole dataset:")
print(type_counts_whole.to_string())
