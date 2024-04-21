import pandas as pd

df = pd.read_csv('MataFlt.csv')

# Modify the values in the 'Type' column
df['Type'] = df['Type'].apply(lambda s: s[0] if '0' not in s else 'S0')

df.to_csv('MataSmp.csv', index=False)