# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,accuracy_score

# %% [markdown]
# Set visualiation style

# %%
sns.set_style("whitegrid")

# %% [markdown]
# Data Loading

# %%
DATASET_PATH = "database.csv"
try:
    df = pd.read_csv("database.csv")
except Exception as e:
    print(f"Error loading CSV: {e}")


df

# %%
# Define a function to clean numerical columns that use commas as decimal separators
def clean_numeric_columns(series):
    # Check if the series is of object type (string)
    if series.dtype == "object":
        # Replace the comma decimal separator with a dot and convert to float
        return series.str.replace(",",".",regex=True).astype(float)
    return series

# %%



