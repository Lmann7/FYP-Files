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
from sklearn.impute import SimpleImputer

# Load the data from the output CSV file
df = pd.read_csv('Mata-P3D-2.6-S2.csv')

# Initialize the imputer with mean strategy
imp = SimpleImputer(strategy='median')

# Dimensions for inputs
in1 = 4
in2 = 165

# Prepare X and y for model training
x = df.iloc[1:, in1:in2].values   # Input features
x = pd.DataFrame(imp.fit_transform(x))

df.iloc[1:, in1:in2] = x

df.to_csv('Mata-P3D-2.6-S2-imputed.csv', index=False)