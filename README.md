# ML Traffic Classification (ISCX VPN-nonVPN 2016)

Machine learning project to classify network traffic into application types using **time-based flow features**,
and to tell apart **normal** traffic from traffic sent through a **VPN**.

Each team member trains and evaluates **one model** on the same processed data, so all models can be compared fairly.

---

## Team

| Member | Name | GitHub | Responsibility | Status |
|---|---|---|---|---|
| Member 1 | _(add name)_ | _(add handle)_ | Data inspection, cleaning and processed train/test files | Done |
| Member 2 | _(add name)_ | _(add handle)_ | Logistic Regression | Done |
| Member 3 | _(add name)_ | _(add handle)_ | XGBoost | Done |
| Member 4 | _(add name)_ | _(add handle)_ | _(add model)_ | _(add status)_ |
| Member 5 | _(add name)_ | _(add handle)_ | _(add model)_ | _(add status)_ |

> Each member: please fill in your own row and your own section below.

---

## Dataset

- **Source:** ISCX VPN-nonVPN 2016, **Scenario B** (time-based features), ARFF files in `data/raw/Scenario_B/`
- **Time windows (timeouts):** 15s, 30s, 60s, 120s
- **Tasks:**
  - `14class`: 7 traffic types x (normal / VPN), e.g. `CHAT`, `VPN-CHAT`, `VOIP`, `VPN-VOIP`
  - `7class`: the "AllinOne" files with 7 traffic types (`BROWSING`, `CHAT`, `FT`, `MAIL`, `P2P`, `STREAMING`, `VOIP`)
- **Features:** 21 time-based features that are common to all four timeouts

```
duration, min_fiat, min_biat, max_fiat, max_biat, mean_fiat, mean_biat,
flowPktsPerSecond, flowBytesPerSecond, min_flowiat, max_flowiat, mean_flowiat, std_flowiat,
min_active, mean_active, max_active, std_active, min_idle, mean_idle, max_idle, std_idle
```

### Data preparation (Member 1)

Done by `src/build_processed.py`. The original ARFF files are never changed.

1. Keep only the 21 common features (the 60s files use different `fiat`/`biat` columns, so only the shared ones are used).
2. Remove extra exact duplicate rows (same features and same label).
3. Give rows with identical features the same `group_id`.
4. Flag rows where the same features appear with more than one label (`conflict_flag = 1`). These rows are kept.
5. Make a **group-aware, stratified 80/20 train/test split** (`StratifiedGroupKFold`, seed 42),
   so identical feature vectors never appear in both train and test (`group_overlap = 0`).

Output: `data/processed/{task}_{timeout}s_train.csv` and `..._test.csv`, plus `processing_summary.csv`.

| Timeout | Task | Raw rows | After dedup | Train | Test |
|---|---|---:|---:|---:|---:|
| 15s | 14class | 18,758 | 18,074 | 14,459 | 3,615 |
| 30s | 14class | 14,651 | 13,638 | 10,910 | 2,728 |
| 60s | 14class | 15,515 | 14,729 | 11,783 | 2,946 |
| 120s | 14class | 10,782 | 9,589 | 7,671 | 1,918 |

Investigations behind these choices (negative values, `min_flowiat`, duplicates, normal vs AllinOne, schema checks)
are in `src/investigations/`, and summary reports are in `reports/` (`dataset_audit_summary.csv`, `class_distributions.csv`).

**Main data notes**

- The data is **imbalanced** (for example `MAIL` has far fewer rows than `VOIP` or `BROWSING`),
  so **macro F1** is used as the main score, together with balanced accuracy and MCC.
- Some features have `-1` values. These were investigated and kept.

---

## Project structure

```
ML-Traffic-Classification/
├── data/
│   ├── raw/Scenario_B/            original ARFF files (15s, 30s, 60s, 120s + AllinOne)
│   └── processed/                 cleaned train/test CSV files for every timeout and task
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   └── 02_xgboost_model.ipynb
├── reports/
│   ├── class_distributions.csv
│   ├── dataset_audit_summary.csv
│   ├── figures/                   Logistic Regression confusion matrices
│   └── results/
│       ├── logreg_*.csv, logistic_regression_*.csv    Logistic Regression results
│       └── xgboost/                XGBoost results (figures, tables, model info)
├── src/
│   ├── load_data.py               shared ARFF loader
│   ├── build_processed.py         builds data/processed/
│   ├── preprocessing/             dataset audit
│   ├── investigations/            data quality checks
│   └── models/                    model scripts
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11 was used. To rebuild the processed data (optional, the files are already in the repo):

```bash
cd src
python build_processed.py
```

---

## Models

All models use the same processed files from `data/processed/`, the same 21 features and the same metrics,
so the results can be compared directly. The main development dataset is **14class, 15s**.

### Logistic Regression (Member 2)

- Scripts: `src/models/logistic_regression.py`, `src/models/logistic_regression_all_timeouts.py`
- Pipeline: `StandardScaler` + `LogisticRegression`, two versions: **baseline** and **balanced** (`class_weight="balanced"`)
- Run on all four timeouts (14class)
- Results: `reports/results/logistic_regression_timeout_comparison.csv`, per-run reports `reports/results/logreg_*.csv`,
  confusion matrices in `reports/figures/`

| Timeout | Version | Accuracy | Macro F1 | Balanced acc. | MCC |
|---|---|---:|---:|---:|---:|
| 15s | baseline | 0.481 | 0.295 | 0.308 | 0.414 |
| 15s | balanced | 0.431 | 0.349 | 0.407 | 0.387 |
| 30s | balanced | 0.353 | 0.324 | 0.395 | 0.316 |
| 60s | balanced | 0.362 | 0.280 | 0.320 | 0.316 |
| 120s | balanced | 0.322 | 0.285 | 0.347 | 0.267 |

_(Member 2: add your notes and conclusions here.)_

### XGBoost (Member 3)

- Notebook: `notebooks/02_xgboost_model.ipynb` (markdown explanation in every section)
- Results folder: `reports/results/xgboost/` (`figures/`, `tables/`, `model/`, `best_params.json`)
- Needs the `xgboost` package (already in `requirements.txt`). The full notebook takes about 45 minutes on a 12-core laptop.

**What the notebook does**

1. Labels encoded with `LabelEncoder` (XGBoost needs class numbers)
2. Baseline `XGBClassifier` (default settings, `tree_method="hist"`)
3. **Hyperparameter tuning** with `RandomizedSearchCV` (40 settings) using **group-aware 5-fold cross validation** (`StratifiedGroupKFold` on `group_id`), best model picked by macro F1
4. Class weight check: tuned model with and without balanced sample weights (chosen by CV)
5. Cross validation of the best model (score of every fold)
6. **Learning curves** (baseline vs tuned)
7. **Boosting curve**: train and validation log loss / error after every tree
8. **Validation curves** for `n_estimators`, `max_depth`, `learning_rate`
9. Final test on the hold-out test set (used only once)
10. Per-class report, confusion matrix, most common mistakes
11. Gain and permutation feature importance
12. Same tuned settings on all four timeouts, and comparison with Logistic Regression

**Best settings:** `n_estimators=400`, `max_depth=10`, `learning_rate=0.2`, `subsample=0.6`, `colsample_bytree=0.7`,
`min_child_weight=3`, `gamma=0`, `reg_lambda=5`, `reg_alpha=0.5`, no class weights

**Results (14class, 15s)**

| | Accuracy | Macro F1 | Balanced acc. | MCC |
|---|---:|---:|---:|---:|
| 5-fold CV (train) | 0.850 | 0.822 ± 0.008 | 0.819 | |
| Test (baseline, default settings) | 0.854 | 0.816 | 0.811 | 0.837 |
| **Test (tuned)** | **0.861** | **0.824** | **0.821** | **0.844** |

| Timeout | Accuracy | Macro F1 | MCC |
|---|---:|---:|---:|
| 15s | 0.861 | 0.824 | 0.844 |
| 30s | 0.850 | 0.814 | 0.831 |
| 60s | 0.867 | 0.823 | 0.851 |
| 120s | 0.818 | 0.773 | 0.785 |

**Key findings**

- Macro F1 is about **2.4x higher** than Logistic Regression (0.824 vs 0.349 on 15s), because the class boundaries are non-linear.
- Tuning gave a small but consistent gain over the default XGBoost (CV macro F1 0.819 → 0.822, test 0.816 → 0.824).
- Balanced class weights raised balanced accuracy a little but lowered macro F1, so they were not used.
- The model still has a train–validation gap (train macro F1 ≈ 0.98, validation ≈ 0.82), and the learning curve is still rising, so **more data would help**.
- The boosting curve shows validation log loss is lowest around 180 trees, while macro F1 keeps improving slowly up to 400 trees.
- Hardest pairs: `BROWSING` ↔ `VPN-BROWSING`, `VPN-CHAT` ↔ `VPN-FT`, `VPN-P2P` → `P2P`. Easiest classes: `VOIP`, `VPN-VOIP` (F1 ≈ 0.99).
- Most important features (permutation): `flowBytesPerSecond`, `min_flowiat`, `duration`, `max_biat`.
- Shorter windows work better; 120s is the weakest (fewest training rows).

### _(Model name)_ (Member 4)

- Notebook / script: _(add path)_
- Results folder: _(add path)_
- Method: _(tuning, cross validation, etc.)_

| | Accuracy | Macro F1 | Balanced acc. | MCC |
|---|---:|---:|---:|---:|
| CV | | | | |
| Test | | | | |

_(Add your notes and conclusions here.)_

### _(Model name)_ (Member 5)

- Notebook / script: _(add path)_
- Results folder: _(add path)_
- Method: _(tuning, cross validation, etc.)_

| | Accuracy | Macro F1 | Balanced acc. | MCC |
|---|---:|---:|---:|---:|
| CV | | | | |
| Test | | | | |

_(Add your notes and conclusions here.)_

---

## Model comparison (14class, 15s, test set)

| Model | Member | Accuracy | Macro F1 | Balanced acc. | MCC |
|---|---|---:|---:|---:|---:|
| Logistic Regression (baseline) | Member 2 | 0.481 | 0.295 | 0.308 | 0.414 |
| Logistic Regression (balanced) | Member 2 | 0.431 | 0.349 | 0.407 | 0.387 |
| XGBoost (tuned) | Member 3 | 0.861 | 0.824 | 0.821 | 0.844 |
| _(add model)_ | Member 4 | | | | |
| _(add model)_ | Member 5 | | | | |

---

## Evaluation rules (please follow for new models)

- Use the files in `data/processed/` only. Do not re-split the data.
- Use the 21 common features and the `label` column as the target.
- Use the **training set only** for tuning and cross validation. Use `group_id` with `StratifiedGroupKFold` to avoid leakage.
- Use the **test set once**, for the final result.
- Report at least: accuracy, macro F1, balanced accuracy, MCC.
- Save results in your own folder: `reports/results/<model_name>/`.
- Work on your own branch (for example `member4-<model>`) and open a pull request into `main`.

---

## Conclusion

_(To be written by the team after all models are finished.)_
