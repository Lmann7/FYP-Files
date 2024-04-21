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
df = pd.read_csv('hubble-in-E+S.csv')
df = df[df['gz2_class'] != 'A']
# Prepare X and y for model training
x = df.iloc[1:, 1:9].values   # Input features (columns 2-9)

# Convert string labels into numerical form
le = LabelEncoder()
y = le.fit_transform(df.loc[1:, 'gz2_class'].values)

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
                   solver='liblinear', tol=0.000001, verbose=0, warm_start=False)   
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
#y_pred_all = model.predict(scaler.transform(x))  # Standardises the original x set and predicts the labels

# Convert the numerical labels back to string labels
y_pred_multi = le.inverse_transform(y_pred)  # Inverse transform the labels

# Create a new dataframe with the input x's and the predicted y's
#df_new = pd.DataFrame({'Log(N/Ha)': x_testold[:, 0], 'Log(O/Hb)': x_testold[:, 1], 'EW_peak_ha_6564': x_testold[:, 2], 'Predicted': y_pred_multi})

# Save the new dataframe to a csv file called 'wholeoutput.csv'
#df_new.to_csv('TestOutput.csv', index=False)  # Save the dataframe to a csv file without row index

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
for i in range(2):
    for j in range(2):
        if i == 0:
            ax.text(j, i, f"{cm[i, j]}", ha='center', va='center', color='white', fontsize=20, 
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
        elif i == 1:
            ax.text(j, i, f"{cm[i, j]}", ha='center', va='center', color='white', fontsize=20, 
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
plt.show()

print(model.coef_)