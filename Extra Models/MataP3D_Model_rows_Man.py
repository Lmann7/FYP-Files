import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd

# Load the data from the output CSV file
df = pd.read_csv('Mata-P3D-Manual-rows.csv')
df = df[(df['Type'] != 'I') & (df['Type'] != 'B') & (df['Type'] != 'd')]

# Prepare X and y for model training
x = df.iloc[1:, 3:202].values   # Input features (columns 2-3)

# Convert string labels into numerical form
le = LabelEncoder()
y = le.fit_transform(df.loc[1:, 'Type'].values)   # Target variable ('Classification' column)

# Print out the mapping of classes
print("String to numerical label mapping:")
for i, class_name in enumerate(le.classes_):
    print(f"{class_name}: {i}")

x_train, x_test, y_train, y_test =\
    train_test_split(x, y, test_size=0.2, random_state=0)   # Splits data into training and testing sets

scaler = StandardScaler()       
x_train = scaler.fit_transform(x_train)                     # Standardises the inputs

model = LogisticRegression(C=10e5, class_weight=None, dual=False, fit_intercept=True,   # Defining model # multi_class='ovr'
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='ovr', n_jobs=None, penalty='l2', random_state=0,
                   solver='liblinear', tol=0.001, verbose=0, warm_start=False)   
model.fit(x_train, y_train)            # Trains model

x_testold = x_test
x_test = scaler.transform(x_test)      # Standardises the testing set inputs
y_pred = model.predict(x_test)         # Uses the model to predict y_train classifications

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
            ax.text(j, i, f"{cm[i, j]}/89", ha='center', va='center', color='white', fontsize=20, 
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
        elif i == 1:
            ax.text(j, i, f"{cm[i, j]}/379", ha='center', va='center', color='white', fontsize=20, 
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
        elif i == 2:
            ax.text(j, i, f"{cm[i, j]}/406", ha='center', va='center', color='white', fontsize=20, 
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
        else:
            ax.text(j, i, f"{cm[i, j]}/188", ha='center', va='center', color='white', fontsize=20, 
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
plt.show()

#print(model.coef_)

import numpy as np

# Get the coefficients from the trained model
coefficients = model.coef_

# Get the column names from the DataFrame
column_names = df.columns[3:202]

# Create a DataFrame from the coefficients
coef_df = pd.DataFrame(coefficients, columns=column_names)

# The width of each bar
width = 0.2

# Create an array with the positions of each bar on the x-axis
x_pos = np.arange(len(coef_df.columns))

# Create a single bar chart
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(x_pos, coef_df.iloc[0], height=width)  # Use barh for horizontal bars
plt.title('Coefficients for Model')
plt.xlabel('Coefficient')
plt.ylabel('Input variable')
plt.yticks(x_pos, coef_df.columns)  # Set y-axis labels
plt.show()








