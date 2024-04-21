import pandas as pd

df1 = pd.read_csv('merge with extra fluxes.csv')
df2 = pd.read_csv('SQL search - Initial fluxes.csv')

merged_df = pd.merge(df1, df2, on='plateifu', how='inner')

merged_df.to_csv('allfluxes_in-simplified-with repeat.csv', index=False)