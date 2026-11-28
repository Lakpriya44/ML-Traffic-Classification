
from pathlib import Path
import pandas as pd
import numpy as np

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Processed dataset directory
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Select the 15-second 14-class datasets
TRAIN_FILE = DATA_DIR / "14class_15s_train.csv"
TEST_FILE = DATA_DIR / "14class_15s_test.csv"

# The 21 numerical features selected during preprocessing
FEATURES = [
    "duration", "min_fiat", "min_biat",
    "max_fiat", "max_biat", "mean_fiat",
    "mean_biat", "flowPktsPerSecond",
    "flowBytesPerSecond", "min_flowiat",
    "max_flowiat", "mean_flowiat", "std_flowiat",
    "min_active", "mean_active", "max_active",
    "std_active", "min_idle", "mean_idle",
    "max_idle", "std_idle"
]

TARGET = "label"

# Load training and testing datasets
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Separate input features (X) and target labels (y)
X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# Display dataset information
print("Training features:", X_train.shape)
print("Training labels:", y_train.shape)

print("Testing features:", X_test.shape)
print("Testing labels:", y_test.shape)

print("Number of classes:", y_train.nunique())

print("Training missing values:", X_train.isnull().sum().sum())
print("Testing missing values:", X_test.isnull().sum().sum())

print("Training conflict records:", train_df["conflict_flag"].sum())
print("Testing conflict records:", test_df["conflict_flag"].sum())




# Import Machine Learning tools
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Create the ML pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

# Train the model
model.fit(X_train, y_train)

# Calculate training accuracy
train_accuracy = model.score(X_train, y_train)

# Calculate testing accuracy
test_accuracy = model.score(X_test, y_test)

print("\n===== LOGISTIC REGRESSION RESULTS =====")
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)



from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# Generate predictions using the trained model
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
balanced_acc = balanced_accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy:           {accuracy:.4f}")
print(f"Macro-F1:           {macro_f1:.4f}")
print(f"Balanced Accuracy:  {balanced_acc:.4f}")
print(f"MCC:                {mcc:.4f}")

# Detailed performance for each traffic class
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred, zero_division=0))

# Create confusion matrix
class_labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=class_labels)

plt.figure(figsize=(12, 9))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title("Logistic Regression - 15s - 14 Classes")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# Create the figures directory
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Save the confusion matrix
plt.savefig(
    FIGURES_DIR / "logreg_14class_15s_baseline_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix saved successfully!")



# Experiment 2: Logistic Regression with balanced class weights

balanced_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced"
    ))
])

# Train the balanced model
balanced_model.fit(X_train, y_train)

# Predict test labels
balanced_pred = balanced_model.predict(X_test)

# Evaluate
print("\n===== BALANCED LOGISTIC REGRESSION =====")

print(
    "Accuracy:",
    accuracy_score(y_test, balanced_pred)
)

print(
    "Macro-F1:",
    f1_score(y_test, balanced_pred, average="macro")
)

print(
    "Balanced Accuracy:",
    balanced_accuracy_score(y_test, balanced_pred)
)

print(
    "MCC:",
    matthews_corrcoef(y_test, balanced_pred)
)



# Detailed evaluation of the balanced model
print("\n===== BALANCED CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        balanced_pred,
        zero_division=0
    )
)

# Create balanced model confusion matrix
class_labels = sorted(y_test.unique())

balanced_cm = confusion_matrix(
    y_test,
    balanced_pred,
    labels=class_labels
)

plt.figure(figsize=(12, 9))

sns.heatmap(
    balanced_cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title("Balanced Logistic Regression - 15s - 14 Classes")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# Save the confusion matrix
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    FIGURES_DIR / "logreg_14class_15s_balanced_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nBalanced confusion matrix saved successfully!")



import joblib

# Create output directories
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Compare both experiments
experiments = [
    ("Baseline", model, y_pred),
    ("Balanced", balanced_model, balanced_pred)
]

results = []

for experiment_name, fitted_model, predictions in experiments:

    results.append({
        "experiment": experiment_name,
        "timeout": 15,
        "task": "14class",
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test, predictions, average="macro"
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "mcc": matthews_corrcoef(y_test, predictions)
    })

# Save comparison results
results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS_DIR / "logistic_regression_15s_comparison.csv",
    index=False
)

# Save trained models
joblib.dump(
    model,
    MODELS_DIR / "logistic_regression_15s_baseline.joblib"
)

joblib.dump(
    balanced_model,
    MODELS_DIR / "logistic_regression_15s_balanced.joblib"
)

print("\n===== SAVED PROJECT OUTPUTS =====")
print(results_df.to_string(index=False))
print("\nEvaluation results and models saved successfully!")
