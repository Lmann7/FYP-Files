import pandas as pd

# Read the original CSV file
df= pd.read_csv('hubble-in.csv')
# Modify the values in the desired column
df['gz2_class'] = df['gz2_class'].str[0]  # Extract the first letter

# Save the modified DataFrame to a new CSV file
df.to_csv('hubble-in-E+S.csv', index=False)

    
