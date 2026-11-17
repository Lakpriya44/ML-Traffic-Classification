"""
duplicate_policy_check.py
Measures exact duplicates and label conflicts on the 21 common features,
for all timeouts and both tasks. Measures only: nothing is saved or deleted.
"""

import pandas as pd
from load_data import load_arff, TIMEOUTS

COMMON = [
    'duration', 'min_fiat', 'min_biat', 'max_fiat', 'max_biat', 'mean_fiat',
    'mean_biat', 'flowPktsPerSecond', 'flowBytesPerSecond', 'min_flowiat',
    'max_flowiat', 'mean_flowiat', 'std_flowiat', 'min_active', 'mean_active',
    'max_active', 'std_active', 'min_idle', 'mean_idle', 'max_idle', 'std_idle',
]


def analyse(df):
    d = df[COMMON + ["label"]].copy()
    n_raw = len(d)

    # Exact duplicates (same features AND same label)
    extra_exact = int(d.duplicated(keep="first").sum())

    # Conflicts: same features, more than one different label (checked on raw rows)
    n_labels = d.groupby(COMMON)["label"].transform("nunique")
    conflict_mask = n_labels > 1
    conflict_records = int(conflict_mask.sum())
    conflict_groups = int(d.loc[conflict_mask, COMMON].drop_duplicates().shape[0])

    # Step A: remove extra exact duplicates only
    a = d.drop_duplicates(keep="first").reset_index(drop=True)

    # Step B: also remove all rows that belong to conflicting groups
    nl = a.groupby(COMMON)["label"].transform("nunique")
    b = a[nl == 1].reset_index(drop=True)

    return d, a, b, {
        "rows_raw": n_raw,
        "extra_exact_dups": extra_exact,
        "conflict_groups": conflict_groups,
        "conflict_records": conflict_records,
        "rows_after_A": len(a),
        "min_class_after_A": int(a["label"].value_counts().min()),
        "rows_after_B": len(b),
        "min_class_after_B": int(b["label"].value_counts().min()),
    }


rows = []
keep = {}
for task in ["14class", "7class"]:
    for t in TIMEOUTS:
        df, _ = load_arff(t, task)
        d, a, b, r = analyse(df)
        rows.append({"timeout": t, "task": task, **r})
        keep[(t, task)] = (d, a, b)

pd.set_option("display.width", 250)
print("\nA = remove extra exact duplicates only")
print("B = A, then also remove every row in a conflicting group\n")
print(pd.DataFrame(rows).to_string(index=False))

# Class-by-class effect at 15s (our development dataset)
for task in ["14class", "7class"]:
    d, a, b = keep[(15, task)]
    table = pd.DataFrame({
        "raw": d["label"].value_counts(),
        "after_A": a["label"].value_counts(),
        "after_B": b["label"].value_counts(),
    }).fillna(0).astype(int)
    print(f"\n=== 15s {task}: class counts ===")
    print(table.to_string())