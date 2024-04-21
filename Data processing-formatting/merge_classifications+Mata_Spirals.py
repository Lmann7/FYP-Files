import pandas as pd

df1 = pd.read_csv('MataSmp-S.csv')
df2 = pd.read_csv('Mata-P3D-2.6.csv')

# remove hyphens and strip whitespace
df1['plateifu'] = df1['plateifu'].str.replace('-', '_').str.strip()
df2['plateifu'] = df2['plateifu'].str.replace('-', '_').str.strip()

merged_df = pd.merge(df1.iloc[:,[0,2]], df2.iloc[:, :], on='plateifu', how='inner')

merged_df.to_csv('Mata-P3D-2.6-S2.csv', index=False)
