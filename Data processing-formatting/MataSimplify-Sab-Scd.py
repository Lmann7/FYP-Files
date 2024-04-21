import pandas as pd

df = pd.read_csv('MataFlt.csv')

# Define the list of special types
#special_types = {
#    'S-merger', 'Irr', 'E(dSph)', 'dSph', 'E+E', 'IrrB', 'SAB-merger',
#    'S0(dwarf)', "BCD", "IrrAB", "dIrr"
#}

# Function to transform the 'Type' column
def transform_type(s):
#    if s in special_types:
#        print(f"Changing '{s}' to 'Other'")  # Add this line
#        return "Other"
    if s.startswith('E'):
        return s[0]
    elif s.startswith('S'):
        if '0' in s:
            return 'S0'
        elif 'ab' in s:
            return 'S-a+b'
        elif 'a' in s:
            return 'S-a+b'
        elif 'bc' in s:
            return 'Sbc'
        elif 'b' in s:
            return 'S-a+b'
        elif 'cd' in s:
            return 'S-c+d'
        elif 'c' in s:
            return 'S-c+d'
        elif 'd' in s:  
            return 'S-c+d'
        else:
            return s
    else:
        return s  # Return the original string if no conditions match

# Apply the transformation
df['Type'] = df['Type'].apply(transform_type)

# Save the modified DataFrame
df.to_csv('MataSmp-S2.csv', index=False)