import pandas as pd

df1 = pd.read_csv('Predictions.csv')
df2 = pd.read_csv('MataFlt.csv')

# remove hyphens and strip whitespace
df1['plateifu'] = df1['plateifu'].str.replace('-', '_').str.strip()
df2['plateifu'] = df2['plateifu'].str.replace('-', '_').str.strip()

merged_df = pd.merge(df1.iloc[:, 0:3], df2.iloc[:, [0, 2]], on='plateifu', how='inner')

merged_df.to_csv('Complete_class.csv', index=False)
