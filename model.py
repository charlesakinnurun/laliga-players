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
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,accuracy_score,f1_score

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
# Select the features we wil use for classification

FEATURES = [
    "Minutes","Goals","Assists","Total Shoot","Expected Goals (xG)",
    "Expected Assists (xAG)","Pass Completion %","Progressive Passes",
    "Carries","Successful Dribbles", "Team" # Team is added as categorical feature
]

TARGET = "Is_Forward"

# %%
# Data Cleaning
# Clean columns with comma separators

for col in ["Expected Goals (xG)","Expected Assists (xAG)","Pass Completion %"]:
    if col in df.columns:
        df[col] = clean_numeric_columns(df[col])

# %%
# Calculate performance metrics per 90 minutes (P90) to normalize for playing time
# We add a small constant (1e-6) to the minutes to avoid division by zero
df["Minutes_Adj"] = df["Minutes"].apply(lambda x : x if x > 0 else 1e-6)

# %%
# Create P90 features for better interpretation
P90_FEATURES = ["Goals_P90","Assists_P90","TotalShoot_P90","xG_P90","xAG_P90"]
df["Goals_P90"] = (df["Goals"] / df["Minutes_Adj"]) * 90
df["Assists_P90"] = (df["Assists"] / df["Minutes_Adj"]) * 90
df["TotalShoot_P90"] = (df["Total Shoot"] / df["Minutes_Adj"]) * 90
df["xG_P90"] = (df["Expected Goals (xG)"] / df["Minutes_Adj"]) * 90
df["xAG_P90"] = (df["Expected Assists (xAG)"] / df["Minutes_Adj"]) * 90

# %%
# Target Variable Creation (Binary Classification)
# We want to predict if a player is a Forward (FW) or not (Not FW)
# The "Position" column often contains multiple values (e.g, "RW", "LW", "FW")
# We check if the "FW" is present 
df[TARGET] = df["Position"].apply(lambda x: 1 if "FW" in str(x) else 0)

# %%
# Filter out players with a very few minutes to reduce noise from small samples
df_clean = df[df["Minutes"] >= 90].copy()

# %%
# Final feature set (using P90 features)
X = df_clean[P90_FEATURES + ["Pass Completion %","Team"]]
y = df_clean[TARGET]

# %% [markdown]
# Data Splitting

# %%
# Split the data into training and testing set (70% train, 30% test)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# %% [markdown]
# Visualization before training

# %%
# Setup for the visualization plot
fig,axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle("Exploratory Data Analysis: Forward (1) vs Non-Forward (0)",fontsize=16)

# Visualization 1: Target Class Distribution
sns.countplot(x=y_train,ax=axes[0],palette="viridis")
axes[0].set_title("Target Class Distribution (Train Set)")
axes[0].set_xlabel("Is Forward (1) / Not Forward (0)")
axes[0].set_ylabel("Count")
# This plot shows how balanced the dataset is 

# Visualization 2: Feature Comparison (Goals_P90)
sns.boxplot(x=TARGET,y="Goals_P90",data=df_clean,ax=axes[0],palette="coolwarm")
axes[1].set_title("Goals P90 Distribution by Class")
axes[1].set_xlabel("Is Forward (1) / Not Forward (0)")
axes[1].set_ylabel("Goals per 90 Minutes")
# This plot confirms that Fowards (1) generally have higher offensive metrics

plt.tight_layout()
plt.show()

# %% [markdown]
# ML Pipeline and Model Comparison

# %%
# Define feature types for preprocessing
numerical_features = P90_FEATURES + ["Pass Completion %"]
categorical_features = ["Team"]

# %%
# Create a preprocessing pipeline using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers = [
        # Apply StandardScaler to numerical features (required for distance-based models like KNN and SVM)
        ("num",StandardScaler(),numerical_features),
        # Apply OneHotEncode to categorical feature (handles non-numeric text data)
        ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features)
    ]
)

# %%
#  Define the models to compare
# We store the models in a dictionary for easy iteration and comparison
models = {
    "Logistic Regression" : LogisticRegression(random_state=42,solver="liblinear"),
    "K-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=5),
    "Support Vector Classifier" : SVC(random_state=42,probability=True),
    "Decision Tree" : DecisionTreeClassifier(random_state=42,max_depth=5),
    "Random Forest" : RandomForestClassifier(random_state=42,n_estimators=100)
}

# %%
# Dictionary to store results
results = {}
best_score = 0
best_model_name = ""
best_pipeline = None

# %% [markdown]
# Model Training and Cross Validation

# %%
# Iterate through each model to train and evaluate
for name,model in models.items():
    # Create a full pipeline: Preprocessing --> Model
    # This ensures that all preprocessing steps (scaling,encoding) are applied consistently
    # Inside the cross-validation loops, preventing data leakage
    pipeline = Pipeline(steps=[("preprocessor",preprocessor), ("classifier",model)])

    # Perform 5-Fold cross-validation on the training data
    # We use F1-Score as it balances Precision and Recall, crucial for clasification tasks
    cv_scores = cross_val_score(pipeline,X_train,y_train,cv=5,scoring="f1")

    # Calculate the mean and standard deviation of the cross-validation scores
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()

    #  Store results
    results[name] = mean_score

    # Check for the best performing model based on mean F1-score
    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name
        best_pipeline = pipeline # Store the best pipeline for final training and prediction

    # Print the performance metrics
    print(f"Model: {name}")
    print(f"  Cross-Validation F1 Score: {mean_score:.4f} (+/- {std_score:.4f})")

# %% [markdown]
# FINAL EVALUATION AND VISUALIZATION

# %%
print("---- FINAL EVALUATION -----")
print(f"The Best Model is: {best_model_name} with F1 Score of {best_score:.4f}")

# %%
# Train the best model pipeline on entire training set
best_pipeline.fit(X_train,y_train)

# %%
# Make prediction on the unseen test set
y_pred = best_pipeline.predict(X_test)

# %%
# Calculate key final metrics
final_accuracy = accuracy_score(y_test,y_pred)
final_f1 = f1_score(y_test,y_pred)
final_roc_auc = roc_auc_score(y_test,best_pipeline.predict_proba(X_test)[:,1])

print(f"{best_model_name} Performance on Test Set:")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"F1 Score: {final_f1:.4f}")
print(f"ROC AUC Score: {final_roc_auc:.4f}")

# %%
# Visualization 3: Model Comparison Bar Plot
plt.figure(figsize=(10,6))
# Convert results dictionary to a Series for plotting
results_series = pd.Series(results).sort_values(ascending=False)
sns.barplot(x=results_series.index,y=results_series.values,palette="plasma",)
plt.title("Classification Model Comparison (Cross-Validated F1-Score)")
plt.xlabel("Model")
plt.ylabel("Mean F1 Score")
plt.xticks(rotation=45,ha="right")
plt.tight_layout()
plt.show()
# This plot visually compares the performance of all tested models

# %%
# Visualization 4: Confusion Matrix for the Best Model
cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8,6))
# Create a heatmap visualiation of the confusion matrix for clear interpretation
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
            xticklabels=["Not Forward (0)", "Forward (1)"],
            yticklabels=["Not Forward (0)","Forward (1)"])
plt.title(f"Confusion Matrix: {best_model_name}")
plt.xlabel("Predicted label")
plt.ylabel("True Label")
plt.show()







# The confusion matrix shows:
# - Top-Left: True Negatives (Correctly predicted non-forwards)
# - Bottom-Right: True Positives (Correctly predicted forwards)
# - Top-Right: False Positives (Incorrectly predicted forwards)
# - Bottom-Left: False Negatives (Incorrectly predicted non-forwards)

# %%
# --------------------------------------------------------------------------------------------------
# PART 6: PREDICTION SIMULATION (USER INPUT)
# --------------------------------------------------------------------------------------------------

def predict_player_position(model_pipeline):
    # This function simulates taking input for a new player and making a prediction.
    print("\n--- SIMULATING PREDICTION FOR A NEW PLAYER ---")

    # Define a sample input for a hypothetical player (e.g., a highly offensive midfield player)
    # The feature values must match the P90 features used during training.
    # Player stats are normalized per 90 mins (e.g., 0.5 Goals P90 means 1 goal every 2 matches)
    new_player_data = {
        'Goals_P90': 0.35,              # High goals P90 (more offensive)
        'Assists_P90': 0.20,            # Decent assists P90
        'TotalShoot_P90': 3.5,          # High shooting volume
        'xG_P90': 0.25,                 # High expected goals P90
        'xAG_P90': 0.15,                # Moderate expected assists P90
        'Pass Completion %': 88.0,      # Good pass completion (typical of a quality player)
        'Team': 'FC Example'            # New team (OneHotEncoder handles unknown teams gracefully if 'handle_unknown=ignore' is set)
    }

    # Convert the dictionary into a Pandas DataFrame, which the pipeline expects
    new_player_df = pd.DataFrame([new_player_data])

    # Make the prediction
    prediction = model_pipeline.predict(new_player_df)[0]
    
    # Get the probability of being a Forward (Class 1)
    prediction_proba = model_pipeline.predict_proba(new_player_df)[0][1]

    # Map the numerical prediction back to a readable label
    predicted_label = "Forward (FW)" if prediction == 1 else "Not a Forward (Not FW)"

    print(f"Input Player Stats (P90): {new_player_data}")
    print(f"-> Predicted Position: {predicted_label}")
    print(f"-> Probability of being a Forward: {prediction_proba:.2f}")

    if prediction == 1:
        print("\nConclusion: Based on their high Goals and Shooting metrics per 90 minutes, the model classifies this player as a Forward.")
    else:
        print("\nConclusion: Although showing good offensive metrics, the balance of stats suggests this player's profile leans closer to a Midfielder or Defender.")


# Run the prediction simulation using the winning model
if best_pipeline:
    predict_player_position(best_pipeline)


