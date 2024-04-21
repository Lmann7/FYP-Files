import pandas as pd
import numpy as np

# Load the data from the CSV file
df = pd.read_csv('Binary_Input.csv')

# Replace '#DIV/0!' with NaN and convert columns to float starting from the second row
df.iloc[1:, 1] = df.iloc[1:, 1].replace('#DIV/0!', float('nan')).astype(float)
df.iloc[1:, 2] = df.iloc[1:, 2].replace('#DIV/0!', float('nan')).astype(float)

# Perform KL line calculation on the second column starting from the second row
df.loc[1:, 'KL_Calc'] = ((0.61/ ((df.iloc[1:, 1]) - 0.47)) + 1.19)   #=(0.61/(S2-0.47))+1.19

# Compares the calculated KL[log(O/Hb)] values with the actual log(O/Hb) values starting from the second row
df.loc[1:, 'comparison'] = df.loc[1:, 'KL_Calc'] < df.iloc[1:, 2]

# Replaces True/False with 'higher'/'lower' starting from the second row
df.loc[1:, 'comparison'] = df.loc[1:, 'comparison'].replace({True: 'higher', False: 'lower'})

# Creates Classification column, and classifies based on the criteria below
for i in range(1, len(df)):
    if df.iloc[i, 3] > 6:
        if df.loc[i, 'comparison'] == 'higher':
            df.loc[i, 'Classification'] = 'AGN'
        else:
            df.loc[i, 'Classification'] = 'SFG'

# Save the output to a new CSV file
df.to_csv('output_Binary.csv', index=False)

# Display the Table of Data
print(df)

# Prepare X and y for model training
x = df.iloc[1:, 1:3].values   # Input features (columns 2-3)
y = df.loc[1:, 'Classification'].values   # Target variable ('Classification' column)


