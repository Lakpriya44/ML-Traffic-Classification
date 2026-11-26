
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"

TIMEOUTS = [15, 30, 60, 120]

TARGET = "label"

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


# Create output directories
for directory in [FIGURES_DIR, RESULTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. TRAIN AND EVALUATE ONE EXPERIMENT
# --------------------------------------------------

def run_experiment(timeout, experiment_name, class_weight):

    print("\n" + "=" * 65)
    print(f"TIMEOUT: {timeout}s | EXPERIMENT: {experiment_name}")
    print("=" * 65)

    # Load processed datasets
    train_file = DATA_DIR / f"14class_{timeout}s_train.csv"
    test_file = DATA_DIR / f"14class_{timeout}s_test.csv"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Training or testing file missing for {timeout}s"
        )

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Validate required columns
    required_columns = FEATURES + [TARGET, "conflict_flag"]

    for column in required_columns:
        if column not in train_df.columns or column not in test_df.columns:
            raise ValueError(f"Required column missing: {column}")

    # Separate features and target labels
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    if X_train.isnull().any().any() or X_test.isnull().any().any():
        raise ValueError("Missing feature values detected!")

    if y_train.isnull().any() or y_test.isnull().any():
        raise ValueError("Missing target labels detected!")

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)
    print("Number of classes:", y_train.nunique())

    print("Training conflict records:", train_df["conflict_flag"].sum())
    print("Testing conflict records:", test_df["conflict_flag"].sum())

    # Build model pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight=class_weight
        ))
    ])

    # Train model using training data only
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    mcc = matthews_corrcoef(y_test, y_pred)

    train_accuracy = model.score(X_train, y_train)

    print("\nEvaluation results:")
    print(f"Training Accuracy:  {train_accuracy:.4f}")
    print(f"Testing Accuracy:   {accuracy:.4f}")
    print(f"Macro-F1:           {macro_f1:.4f}")
    print(f"Balanced Accuracy:  {balanced_acc:.4f}")
    print(f"MCC:                {mcc:.4f}")

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_file = (
        RESULTS_DIR
        / f"logreg_14class_{timeout}s_{experiment_name}_report.csv"
    )

    report_df.to_csv(report_file)

    # Create confusion matrix
    class_labels = sorted(y_test.unique())

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=class_labels
    )

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues" if experiment_name == "baseline" else "Greens",
        xticklabels=class_labels,
        yticklabels=class_labels
    )

    plt.title(
        f"Logistic Regression - {timeout}s - {experiment_name}"
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    figure_file = (
        FIGURES_DIR
        / f"logreg_14class_{timeout}s_{experiment_name}_confusion_matrix.png"
    )

    plt.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Save trained model
    model_file = (
        MODELS_DIR
        / f"logistic_regression_14class_{timeout}s_{experiment_name}.joblib"
    )

    joblib.dump(model, model_file)

    print("\nSaved classification report:", report_file.name)
    print("Saved confusion matrix:", figure_file.name)
    print("Saved trained model:", model_file.name)

    return {
        "timeout": timeout,
        "experiment": experiment_name,
        "train_accuracy": train_accuracy,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc
    }


# --------------------------------------------------
# 3. RUN ALL FOUR TIMEOUTS
# --------------------------------------------------

def main():

    all_results = []

    experiments = [
        ("baseline", None),
        ("balanced", "balanced")
    ]

    for timeout in TIMEOUTS:

        for experiment_name, class_weight in experiments:

            result = run_experiment(
                timeout,
                experiment_name,
                class_weight
            )

            all_results.append(result)

    # Create final comparison table
    results_df = pd.DataFrame(all_results)

    results_file = (
        RESULTS_DIR / "logistic_regression_timeout_comparison.csv"
    )

    results_df.to_csv(results_file, index=False)

    print("\n" + "=" * 75)
    print("FINAL LOGISTIC REGRESSION TIMEOUT COMPARISON")
    print("=" * 75)

    print(results_df.to_string(index=False))

    print("\nResults saved:", results_file)


if __name__ == "__main__":
    main()
