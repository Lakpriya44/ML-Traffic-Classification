"""
build_processed.py
Builds processed train/test CSV files for all timeouts and both tasks.

Policy:
  - 21 common features only
  - remove extra exact duplicates (same features + same label)
  - keep conflicting rows, flagged in 'conflict_flag'
  - group-aware, stratified 80/20 split (identical feature vectors never
    appear in both train and test)
Original ARFF files are never modified.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from load_data import load_arff, TIMEOUTS, PROJECT_ROOT

COMMON = [
    'duration', 'min_fiat', 'min_biat', 'max_fiat', 'max_biat', 'mean_fiat',
    'mean_biat', 'flowPktsPerSecond', 'flowBytesPerSecond', 'min_flowiat',
    'max_flowiat', 'mean_flowiat', 'std_flowiat', 'min_active', 'mean_active',
    'max_active', 'std_active', 'min_idle', 'mean_idle', 'max_idle', 'std_idle',
]
SEED = 42
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build(timeout, task):
    df, _ = load_arff(timeout, task)
    d = df[COMMON + ["label"]].copy()
    rows_raw = len(d)

    # 1) remove extra exact duplicates
    d = d.drop_duplicates(keep="first").reset_index(drop=True)

    # 2) group id: rows with identical features share the same id
    d["group_id"] = d.groupby(COMMON, sort=False).ngroup()

    # 3) conflict flag: 1 if the same features appear with more than one label
    d["conflict_flag"] = (d.groupby("group_id")["label"].transform("nunique") > 1).astype(int)

    # 4) group-aware stratified split: first fold of 5 = ~20% test
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(sgkf.split(d[COMMON], d["label"], groups=d["group_id"]))
    train = d.iloc[train_idx].reset_index(drop=True)
    test = d.iloc[test_idx].reset_index(drop=True)

    # 5) safety check: no group appears in both train and test
    overlap = set(train["group_id"]) & set(test["group_id"])
    assert len(overlap) == 0, "Group leakage detected!"

    train.to_csv(OUT_DIR / f"{task}_{timeout}s_train.csv", index=False)
    test.to_csv(OUT_DIR / f"{task}_{timeout}s_test.csv", index=False)

    return {
        "timeout": timeout,
        "task": task,
        "rows_raw": rows_raw,
        "rows_after_dedup": len(d),
        "train_rows": len(train),
        "test_rows": len(test),
        "conflict_rows_total": int(d["conflict_flag"].sum()),
        "group_overlap": len(overlap),
        "min_class_train": int(train["label"].value_counts().min()),
        "min_class_test": int(test["label"].value_counts().min()),
    }


summary = []
for task in ["14class", "7class"]:
    for t in TIMEOUTS:
        summary.append(build(t, task))

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT_DIR / "processing_summary.csv", index=False)

pd.set_option("display.width", 250)
print(summary_df.to_string(index=False))

# Class balance check for the 15s development datasets
for task in ["14class", "7class"]:
    tr = pd.read_csv(OUT_DIR / f"{task}_15s_train.csv")
    te = pd.read_csv(OUT_DIR / f"{task}_15s_test.csv")
    table = pd.DataFrame({
        "train": tr["label"].value_counts(),
        "test": te["label"].value_counts(),
    })
    table["test_share_%"] = (100 * table["test"] / (table["train"] + table["test"])).round(1)
    print(f"\n=== 15s {task}: train/test class counts ===")
    print(table.to_string())

print("\nColumns in a saved file:")
print(list(pd.read_csv(OUT_DIR / "14class_15s_train.csv", nrows=1).columns))
print("\nFiles saved in:", OUT_DIR)