import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the data from the output CSV file
df = pd.read_csv('output_Binary.csv')

# Prepare X and y for model training
x = df.iloc[1:, 1:3].values   # Input features (columns 2-3)

# Convert string labels into numerical form
le = LabelEncoder()
y = le.fit_transform(df.loc[1:, 'Classification'].values)   # Target variable ('Classification' column)

# Print out the mapping of classes
print("String to numerical label mapping:")
for i, class_name in enumerate(le.classes_):
    print(f"{class_name}: {i}")

x_train, x_test, y_train, y_test =\
    train_test_split(x, y, test_size=0.2, random_state=0)   # Splits data into training and testing sets

scaler = StandardScaler()       
x_train = scaler.fit_transform(x_train)                     # Standardises the inputs

model = LogisticRegression(solver='liblinear', C=0.05, random_state=0)   # Defining model # multi_class='ovr'
model.fit(x_train, y_train)            # Trains model

x_testold = x_test
x_test = scaler.transform(x_test)      # Standardises the testing set inputs
y_pred = model.predict(x_test)         # Uses the model to predict y_train classifications

print(model.score(x_train, y_train) )  # Accuracy of the model for the training set
print(model.score(x_test, y_test) )    # Accuracy of the model for the testing set

cm = confusion_matrix(y_test, y_pred)   # Confusion matrix
print(cm)
print(classification_report(y_test, y_pred))

# Use the model to predict the labels for the original x set
#y_pred_BinAll = model.predict(scaler.transform(x))  # Standardises the original x set and predicts the labels

# Convert the numerical labels back to string labels
#y_pred_BinAll = le.inverse_transform(y_pred_BinAll)  # Inverse transform the labels

y_pred_Bin = le.inverse_transform(y_pred)  # Inverse transform the labels

# Create a new dataframe with the input x's and the predicted y's
df_new = pd.DataFrame({'Log(N/Ha)': x_testold[:, 0], 'Log(O/Hb)': x_testold[:, 1], 'Predicted': y_pred_Bin})

# Save the new dataframe to a csv file called 'wholeoutput.csv'
df_new.to_csv('BinTestOutput.csv', index=False)  # Save the dataframe to a csv file without row index

# Produces graphic of confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm, cmap='Blues')
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted AGNs', 'Predicted SFGs'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual AGNs', 'Actual SFGs'))
ax.set_ylim(1.5, -0.5)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
for i in range(2):
    for j in range(2):
        if i == 0:
            ax.text(j, i, f"{cm[i, j]}/103", ha='center', va='center', color='white', fontsize=18, 
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
        else:
            ax.text(j, i, f"{cm[i, j]}/442", ha='center', va='center', color='white', fontsize=18, 
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
plt.show()