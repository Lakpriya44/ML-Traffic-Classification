"""Member 3: Random Forest baseline experiments.

Uses Member 1's existing processed train/test CSVs without creating or
modifying any split.

Run all eight experiments from the repository root:
    python src/models/random_forest.py

Run one experiment:
    python src/models/random_forest.py --task 7class --timeout 15
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score,
    matthews_corrcoef, precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "random_forest"
RESULT_DIR = ROOT / "results" / "random_forest"
TASKS = ("7class", "14class")
TIMEOUTS = (15, 30, 60, 120)
EXCLUDED_COLUMNS = ("label", "group_id", "conflict_flag")

MODEL_PARAMETERS: dict[str, Any] = {
    "n_estimators": 300,
    "criterion": "gini",
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}


def load_existing_data(task: str, timeout: int):
    """Load the existing split exactly as produced by Member 1."""
    train_path = DATA_DIR / f"{task}_{timeout}s_train.csv"
    test_path = DATA_DIR / f"{task}_{timeout}s_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing processed split: {train_path} or {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    for name, frame in (("train", train), ("test", test)):
        missing = set(EXCLUDED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} file is missing required columns: {sorted(missing)}")

    features = [c for c in train.columns if c not in EXCLUDED_COLUMNS]
    test_features = [c for c in test.columns if c not in EXCLUDED_COLUMNS]
    if features != test_features:
        raise ValueError("Train/test feature columns or their order do not match.")

    X_train, X_test = train[features].copy(), test[features].copy()
    y_train, y_test = train["label"].astype(str), test["label"].astype(str)

    for name, X in (("train", X_train), ("test", X_test)):
        if X.isna().any().any():
            raise ValueError(f"Missing feature values found in {name} data.")
        if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Infinite or non-finite values found in {name} data.")

    overlap = set(train.group_id).intersection(set(test.group_id))
    if overlap:
        raise ValueError(f"Group leakage check failed: {len(overlap)} shared group IDs.")
    if set(y_test) - set(y_train):
        raise ValueError("The test set contains a class absent from the training set.")

    return train, test, X_train, X_test, y_train, y_test, features


def measure_inference(model, X_test):
    """Warm up on 10 rows, then time one complete test-set prediction pass."""
    model.predict(X_test.iloc[:min(10, len(X_test))])
    start = time.perf_counter()
    model.predict(X_test)
    elapsed = time.perf_counter() - start
    return elapsed, elapsed / len(X_test) * 1000.0


def run_experiment(task: str, timeout: int) -> dict[str, Any]:
    print("\n" + "=" * 68, flush=True)
    print(f"RANDOM FOREST | {task.upper()} | {timeout}s", flush=True)
    print("=" * 68, flush=True)

    train, test, X_train, X_test, y_train, y_test, features = load_existing_data(task, timeout)
    print(f"Training rows: {len(X_train):,} | Testing rows: {len(X_test):,}", flush=True)
    print(f"Features: {len(features)} | Classes: {y_train.nunique()} | Train/test group overlap: 0", flush=True)
    print("Using existing Member 1 train/test files; no new split created.", flush=True)

    model = RandomForestClassifier(**MODEL_PARAMETERS)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    first_prediction_seconds = time.perf_counter() - start

    accuracy = accuracy_score(y_test, y_pred)
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    inference_seconds, latency_per_flow_ms = measure_inference(model, X_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"random_forest_{task}_{timeout}s"
    model_path = MODEL_DIR / f"{stem}.joblib"
    joblib.dump(model, model_path, compress=3)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": y_pred,
        "correct": y_test.to_numpy() == y_pred,
    }).to_csv(RESULT_DIR / f"{stem}_predictions.csv", index=False)

    report = classification_report(y_test, y_pred, labels=model.classes_,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(
        RESULT_DIR / f"{stem}_classification_report.csv")

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    pd.DataFrame(cm,
        index=[f"actual_{x}" for x in model.classes_],
        columns=[f"predicted_{x}" for x in model.classes_],
    ).to_csv(RESULT_DIR / f"{stem}_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(12, 10) if len(model.classes_) > 10 else (9, 7))
    ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(
        ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Random Forest – {task} – {timeout}s")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / f"{stem}_confusion_matrix.png", dpi=180)
    plt.close(fig)

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(RESULT_DIR / f"{stem}_feature_importance.csv", index=False)
    top = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.feature, top.importance)
    ax.set_title(f"Random Forest Feature Importance – {task} – {timeout}s")
    ax.set_xlabel("Mean decrease in impurity")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / f"{stem}_feature_importance.png", dpi=180)
    plt.close(fig)

    result = {
        "model": "Random Forest", "task": task, "timeout_seconds": timeout,
        "train_rows": len(X_train), "test_rows": len(X_test),
        "number_of_features": len(features), "number_of_classes": int(y_train.nunique()),
        "accuracy": accuracy, "macro_precision": macro_precision,
        "macro_recall": macro_recall, "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc, "mcc": mcc,
        "training_time_seconds": training_seconds,
        "first_test_prediction_seconds": first_prediction_seconds,
        "full_test_inference_seconds": inference_seconds,
        "mean_latency_per_flow_ms": latency_per_flow_ms,
        "model_size_mb": model_size_mb,
        "model_file": str(model_path.relative_to(ROOT)),
        "random_state": MODEL_PARAMETERS["random_state"],
        "n_estimators": MODEL_PARAMETERS["n_estimators"],
        "max_features": MODEL_PARAMETERS["max_features"],
    }
    metadata = {
        **result,
        "feature_columns": features,
        "excluded_columns": list(EXCLUDED_COLUMNS),
        "model_parameters": MODEL_PARAMETERS,
        "scikit_learn_version": sklearn.__version__,
        "split_source": "existing Member 1 processed train/test CSV files",
        "new_split_created": False,
    }
    (RESULT_DIR / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nEvaluation results", flush=True)
    print(f"Accuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f}", flush=True)
    print(f"Balanced Accuracy: {balanced_acc:.4f} | MCC: {mcc:.4f}", flush=True)
    print(f"Training: {training_seconds:.3f}s | Full-test inference: {inference_seconds:.4f}s", flush=True)
    print(f"Latency/flow: {latency_per_flow_ms:.6f}ms | Model size: {model_size_mb:.2f}MB", flush=True)
    print(f"Saved model: {model_path.relative_to(ROOT)}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["7class", "14class", "all"], default="all")
    parser.add_argument("--timeout", choices=["15", "30", "60", "120", "all"], default="all")
    args = parser.parse_args()
    tasks = TASKS if args.task == "all" else (args.task,)
    timeouts = TIMEOUTS if args.timeout == "all" else (int(args.timeout),)

    results = [run_experiment(task, timeout) for task in tasks for timeout in timeouts]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(results)
    summary.to_csv(RESULT_DIR / "random_forest_summary.csv", index=False)

    columns = ["task", "timeout_seconds", "accuracy", "macro_f1",
               "balanced_accuracy", "mcc", "training_time_seconds",
               "full_test_inference_seconds", "mean_latency_per_flow_ms", "model_size_mb"]
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in summary[columns].iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(vals) + " |")
    (RESULT_DIR / "random_forest_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nFINAL RANDOM FOREST SUMMARY", flush=True)
    print(summary[columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
