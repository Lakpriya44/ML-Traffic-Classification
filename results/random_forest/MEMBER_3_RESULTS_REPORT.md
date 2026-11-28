# Member 3 – Random Forest implementation and results

## What was implemented

Eight Random Forest models were trained: 7-class and 14-class classification for the 15s, 30s, 60s and 120s datasets.
The implementation uses the existing processed train/test CSV files from Member 1. No new train/test split was created.

## Input data and safeguards

- 21 numerical flow-feature columns are used as predictors.
- `label` is the target column.
- `group_id` and `conflict_flag` are excluded from model inputs.
- The script checks feature-column consistency, missing values, non-finite values, test-class coverage and train/test group overlap.
- All eight experiments had zero overlapping group IDs between their existing train and test partitions.
- Random Forest does not require feature scaling, so the processed numerical features are used without standardization.

## Random Forest baseline configuration

- `n_estimators=300`
- `criterion="gini"`
- `max_features="sqrt"`
- `bootstrap=True`
- `random_state=42`
- `n_jobs=-1`

This is a fixed baseline configuration. Hyperparameter tuning has not yet been performed.

## Evaluation results

| Task | Timeout | Accuracy | Macro-F1 | Balanced accuracy | MCC |
|---|---:|---:|---:|---:|---:|
| 7class | 15s | 0.8992 | 0.8626 | 0.8580 | 0.8733 |
| 7class | 30s | 0.8958 | 0.8570 | 0.8486 | 0.8657 |
| 7class | 60s | 0.8861 | 0.8409 | 0.8337 | 0.8551 |
| 7class | 120s | 0.8772 | 0.8127 | 0.7913 | 0.8218 |
| 14class | 15s | 0.8473 | 0.8111 | 0.8034 | 0.8294 |
| 14class | 30s | 0.8387 | 0.8016 | 0.7956 | 0.8182 |
| 14class | 60s | 0.8439 | 0.8012 | 0.7906 | 0.8250 |
| 14class | 120s | 0.8076 | 0.7650 | 0.7477 | 0.7719 |

## Observations

- The highest 7-class Macro-F1 was 0.8626 on the 15-second dataset.
- The highest 14-class Macro-F1 was 0.8111 on the 15-second dataset.
- The 7-class results were stronger than the corresponding 14-class results in these experiments.
- The 120-second dataset had the lowest Macro-F1 for both classification tasks.
- These are results from separately prepared datasets; they do not prove that timeout duration alone caused the differences.

## Runtime and saved artifacts

Training time, full-test inference time, per-flow latency and model size are recorded in `random_forest_summary.csv`. Timing depends on the hardware and software environment.

Each experiment has a saved model, prediction CSV, classification report, confusion matrix (CSV and PNG), feature-importance CSV and PNG, and metadata JSON.

## Important notes for group integration

- Member 2 must use the same existing train/test files and label tasks for a fair Logistic Regression comparison.
- Member 4 can use `random_forest_summary.csv` for the model comparison table and chart.
- The comparison with Logistic Regression is pending until Member 2 supplies their metrics.
- Impurity-based feature importance describes how the fitted trees used features; it does not establish causation.
- The proposal mentions 23 predictors, but the existing processed files contain 21 common numerical predictors. This implementation follows the supplied processed files without changing Member 1’s work.

## Verification

All eight saved model files were reloaded and their predictions were checked against the stored prediction CSVs; all matched.
