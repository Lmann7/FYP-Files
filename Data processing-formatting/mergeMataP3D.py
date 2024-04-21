import pandas as pd

df1 = pd.read_csv('MataSmp.csv')
df2 = pd.read_csv('PIPE3Dflt2.csv')

# remove hyphens and strip whitespace
df1['plateifu'] = df1['plateifu'].str.replace('-', '_').str.strip()
df2['plateifu'] = df2['plateifu'].str.replace('-', '_').str.strip()

merged_df = pd.merge(df1.iloc[:, [0, 2]], df2.iloc[:, 0:], on='plateifu', how='inner')

merged_df.to_csv('Mata-P3D-2.csv', index=False)
