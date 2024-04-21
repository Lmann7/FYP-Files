import pandas as pd

df1 = pd.read_csv('Complete_class.csv')
df2 = pd.read_csv('Mata-P3D-2.5.csv')

# Filter rows where column 2 is not equal to column 3 (corrected indexing)
df1 = df1[df1['Predicted_Class'] != df1['Actual_Class']]

# remove hyphens and strip whitespace
df1['plateifu'] = df1['plateifu'].str.replace('-', '_').str.strip()
df2['plateifu'] = df2['plateifu'].str.replace('-', '_').str.strip()

merged_df = pd.merge(df1.iloc[:, 0:3], df2.iloc[:,:], on='plateifu', how='inner')

merged_df.to_csv('Complete-misclass-2.5.csv', index=False)
