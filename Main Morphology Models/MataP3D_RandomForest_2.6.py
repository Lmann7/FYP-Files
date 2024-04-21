import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
from sklearn.impute import SimpleImputer
# Load the data from the output CSV file
df = pd.read_csv('Mata-P3D-2.6-imputed.csv')
df = df[(df['Type'] != 'I') & (df['Type'] != 'B') & (df['Type'] != 'd')]

# Dimensions for input
in1 = 3
in2 = 164
Num_est = 200
split= 0.2
RNDM= 0
print('Number of Estimators:', Num_est)
print('Test Split:', split)
print('Random State:', RNDM)

# Extract plateifu values as a separate DataFrame
plateifu_df = df[['plateifu']]  # Keep only the 'plateifu' column

# Prepare X and y for model training
x = df.iloc[0:, in1:in2].values   # Input features (columns 2-3)

# Swap labels temporarily
#df.loc[df['Type'] == 'E', 'Type'] = 'TEMP'
#df.loc[df['Type'] == 'S', 'Type'] = 'E'
#df.loc[df['Type'] == 'TEMP', 'Type'] = 'S'

# Convert string labels into numerical form
le = LabelEncoder()
y = le.fit_transform(df.loc[0:, 'Type'].values)   # Target variable ('Classification' column)

# Print out the mapping of classes
print("String to numerical label mapping:")
for i, class_name in enumerate(le.classes_):
    print(f"{class_name}: {i}")

# Split data for training and testing
x_train, x_test, y_train, y_test, plateifu_train, plateifu_test = train_test_split(
    x, y, plateifu_df, test_size=split, random_state=RNDM
)

scaler = StandardScaler()  
x_train = scaler.fit_transform(x_train)                     # Standardises the inputs

model = RandomForestClassifier(n_estimators=Num_est, random_state=RNDM) 
model.fit(x_train, y_train)   

x_testold = x_test
x_test = scaler.transform(x_test)      # Standardises the testing set inputs
y_pred = model.predict(x_test)         # Uses the model to predict y_train classifications
y_pred_proba = model.predict_proba(x_test)  # Probabilities of classifications

print(model.score(x_train, y_train) )  # Accuracy of the model for the training set
print(model.score(x_test, y_test) )    # Accuracy of the model for the testing set

cm = confusion_matrix(y_test, y_pred)   # Confusion matrix
print(cm)
print(classification_report(y_test, y_pred))

# Produces graphic of confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(cm,)
ax.grid(False)
ax.set_xlabel('Predicted outputs', fontsize=20, color='black')
ax.set_ylabel('Actual outputs', fontsize=20, color='black')

class_names = le.classes_
ax.xaxis.set(ticks=range(len(class_names)), ticklabels=class_names)
ax.yaxis.set(ticks=range(len(class_names)), ticklabels=class_names)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
for i in range(3):
    for j in range(3):
        if i == 0:
            ax.text(j, i, f"{cm[i, j]}", ha='center', va='center', color='white', fontsize=20, 
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
        elif i == 1:
            ax.text(j, i, f"{cm[i, j]}", ha='center', va='center', color='white', fontsize=20, 
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
        elif i == 2:
            ax.text(j, i, f"{cm[i, j]}", ha='center', va='center', color='white', fontsize=20, 
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
        else:
            ax.text(j, i, f"{cm[i, j]}", ha='center', va='center', color='white', fontsize=20, 
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
plt.show()

# Get the coefficients from the trained model
coefficients = model.feature_importances_

# Get the column names from the DataFrame
column_names = df.columns[in1:in2]

# Create a DataFrame with a single row 
coef_df = pd.DataFrame([coefficients], columns=column_names) 

############################################################################################################
# Combine plateifu values, predicted classifications, and actual classes
combined_df = pd.DataFrame({
    "plateifu": plateifu_test.values.flatten(),
    "Predicted_Class": le.inverse_transform(y_pred),
    "Actual_Class": le.inverse_transform(y_test),  # Include actual classes
    "Probability": np.max(y_pred_proba, axis=1)  # Include probability of predicted class
})

#print(combined_df.head())  # View the combined DataFrame

# Save the new dataframe to a csv file
combined_df.to_csv('RF-Pred-2.6.csv', index=False)  # Save the dataframe to a csv file without row index

########################################################################################################
# Create a figure
fig, ax = plt.subplots(figsize=(8, len(coef_df.columns)*.5))

row_averages = coef_df.abs().mean(axis=0)  # Calculate absolute means across rows
coef_df.loc['temp_sort_column'] = row_averages # Add avearges as a row
coef_df_sorted = coef_df.sort_values(by='temp_sort_column', ascending=True, axis=1)  # Descending sort
# Exclude 'temp_sort_column'
coef_df_sorted = coef_df_sorted.drop('temp_sort_column', axis=0) 
coef_df = coef_df.drop('temp_sort_column', axis=0) 

# Sort coefficients and column names together
#coef_df_sorted = coef_df.sort_values(by=0, ascending=True, axis=1)  

# Set the bar width
bar_width = 0.02

# Set the space between the bars
space = 0.01

# Set the space between the lines for different column titles
line_space = 0.25  # Halve this value to decrease the space between the groups of bars

# Get the number of columns
numb_columns = len(coef_df_sorted.columns)

# Create an index for each column with added line_space
index = np.arange(numb_columns) * line_space

# Create the bars with specified colors
for i in range(len(coef_df_sorted)):
    label = 'Importance'
    color = 'blue' 
    ax.barh(index + i * (bar_width + space), coef_df_sorted.iloc[i], bar_width, label=label, color=color)
    
# Set the y ticks location
ax.set_yticks(index + bar_width / 2)
ax.set_yticklabels(coef_df_sorted.columns, fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.legend(fontsize=25)

plt.title('Coefficients for Model')
plt.xlabel('Coefficient')
plt.ylabel('Input variable')

plt.show()
########################################################################################################
# Sort the DataFrame by coefficient values (descending order)
coef_df_sorted2 = coef_df.T.sort_values(by=0, ascending=False)

coef_df_sorted2.to_csv('RF-importances-2.6.csv', index=True)

# Filter columns based on absolute coefficient values greater than 0.5
selected_columns = coef_df_sorted.columns[(abs(coef_df_sorted.iloc[0]) > 0.01)]

Highest_coef = coef_df_sorted[selected_columns]

# Save the selected measurables and their coefficients to a CSV file
Highest_coef.to_csv('RF-selected_measurables-2.6.csv', index=False)

######################################################################################################
# Create a figure and a set of subplots with a larger size
fig, ax = plt.subplots(figsize=(8, len(Highest_coef.columns)*.8))

# Set the bar width
bar_width = 0.02

# Set the space between the bars
space = 0.01

# Set the space between the lines for different column titles
line_space = 0.25  # Halve this value to decrease the space between the groups of bars

# Get the number of columns
num_columns = len(Highest_coef.columns)

# Create an index for each column with added line_space
index = np.arange(num_columns) * line_space

# Create the bars with specified colors
for i in range(len(Highest_coef)):
    label = 'Importance'
    color = 'blue' 
    ax.barh(index + i * (bar_width + space), Highest_coef.iloc[i], bar_width, label=label, color=color)
    
# Set the y ticks location
ax.set_yticks(index + bar_width / 2)
ax.set_yticklabels(Highest_coef.columns, fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.legend(fontsize=25)

# Show the plot
plt.show()

################################################################################################
from sklearn.inspection import permutation_importance

result = permutation_importance(model, x_test, y_test, n_repeats=10, random_state=42)

feature_importance = pd.DataFrame({'Feature': column_names,
                                   'Importance': result.importances_mean,
                                   'Standard Deviation': result.importances_std})
feature_importance = feature_importance.sort_values('Importance', ascending=True)

ax = feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, len(feature_importance)*.5), xerr='Standard Deviation', capsize=4)
ax.set_xlabel('Permutation Importance')
ax.set_title('Permutation Importance with Standard Deviation')

filtered_importance = feature_importance[feature_importance['Importance'] > .0035]

ax = filtered_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, len(filtered_importance)*.5), xerr='Standard Deviation', capsize=4)
ax.set_xlabel('Permutation Importance')
ax.set_title('Permutation Importance with Standard Deviation')

