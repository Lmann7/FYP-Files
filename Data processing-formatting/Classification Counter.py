import pandas as pd

# Read the CSV data
df = pd.read_csv('Complete_class.csv')

# Count occurrences in the entire dataset (unchanged)
type_counts_whole = df['Type'].value_counts()

def count_type_with_condition(df):
  """
  This function counts the occurrences of each variable in column 'Type'
  of the DataFrame 'df', considering only rows where column 2 is not equal to column 3.

  Args:
      df (pandas.DataFrame): The DataFrame containing the data.

  Returns:
      pandas.Series: A Series containing the counts for each unique value in 'Type'.
  """

  # Filter rows where column 2 is not equal to column 3 (corrected indexing)
  filtered_df = df[df['Predicted_Class'] != df['Actual_Class']]

  # Count occurrences in the filtered DataFrame
  type_counts = filtered_df['Type'].value_counts()

  return type_counts

# Call the function to get the counts with condition
type_counts = count_type_with_condition(df.copy())  # Avoid modifying the original df

# Print conditional counts and fractions
print("Misclassifications:")
for type_name, conditional_count in type_counts.items():
  whole_count = type_counts_whole.get(type_name, 0)  # Handle missing types
  fraction = conditional_count / whole_count if whole_count > 0 else 0
  print(f"{type_name}: {conditional_count} ({fraction:.4f} of total)")


# Print the whole dataset counts (unchanged)
print("\nCounts in whole dataset:")
print(type_counts_whole.to_string())
