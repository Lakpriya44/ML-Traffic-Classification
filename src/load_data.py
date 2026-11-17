"""
load_data.py
Reusable loader for the ISCX-VPN-NonVPN-2016 Scenario B ARFF files.
Original files are only READ, never modified.

Running this file directly checks all 8 files and prints a report.
"""

from io import StringIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.io import arff

# ---- Paths (project root = the folder above src/) ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "Scenario_B"
ALLINONE_DIR = RAW_DIR / "Allinone"

TIMEOUTS = [15, 30, 60, 120]
TARGET_COLUMN = "class1"


def get_path(timeout, task):
    """task = '14class' (Normal files) or '7class' (AllinOne files)."""
    if task == "14class":
        return RAW_DIR / f"TimeBasedFeatures-Dataset-{timeout}s.arff"
    elif task == "7class":
        return (
            ALLINONE_DIR
            / f"TimeBasedFeatures-Dataset-{timeout}s-AllinOne.arff"
        )
    raise ValueError("task must be '14class' or '7class'")


def load_arff(timeout, task):
    """
    Load one ARFF file and return (dataframe, info).
    - decodes the target from bytes to normal strings
    - removes completely blank rows IN MEMORY only (the file is untouched)
    - the target column is renamed to 'label'
    """
    path = get_path(timeout, task)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    raw_text = path.read_text(encoding="utf-8-sig")
    lines = raw_text.splitlines()

    cleaned_lines = []
    in_data_section = False
    n_blank = 0
    attr_count = 0

    for line in lines:
        stripped = line.strip()

        if not in_data_section:
            if stripped.lower().startswith("@attribute"):
                attr_count += 1
            elif stripped.lower().startswith("@data"):
                in_data_section = True
                cleaned_lines.append("@data")
                continue
            cleaned_lines.append(line)
            continue

        # Inside @data section: skip blank lines, comments, and comma-only records
        if not stripped or stripped.startswith("%"):
            continue
        if re.fullmatch(r",+", stripped):
            n_blank += 1
            continue

        # Convert any empty fields ("") to ARFF missing marker "?" and pad short rows
        fields = [
            f.strip() if f.strip() != "" else "?" for f in stripped.split(",")
        ]
        if len(fields) < attr_count:
            fields.extend(["?"] * (attr_count - len(fields)))

        cleaned_lines.append(",".join(fields))

    data, _meta = arff.loadarff(StringIO("\n".join(cleaned_lines)))
    df = pd.DataFrame(data)
    rows_raw = len(df) + n_blank

    # Decode target from bytes (b'VOIP') to str ('VOIP')
    target = df[TARGET_COLUMN].apply(
        lambda v: (
            v.decode("utf-8").strip() if isinstance(v, bytes) else str(v).strip()
        )
    )
    target = target.replace("?", np.nan)
    features = df.drop(columns=[TARGET_COLUMN]).apply(
        pd.to_numeric, errors="coerce"
    )

    out = features.copy()
    out["label"] = target

    info = {"rows_raw": rows_raw, "blank_rows_dropped": n_blank}
    return out, info


def check_all():
    reports = []
    feature_sets = {}

    for task in ["14class", "7class"]:
        for t in TIMEOUTS:
            df, info = load_arff(t, task)
            feats = [c for c in df.columns if c != "label"]
            feature_sets[(t, task)] = feats
            arr = df[feats].to_numpy(dtype="float64")

            reports.append(
                {
                    "timeout": t,
                    "task": task,
                    "rows_raw": info["rows_raw"],
                    "blank_dropped": info["blank_rows_dropped"],
                    "rows_final": len(df),
                    "n_features": len(feats),
                    "missing": int(np.isnan(arr).sum()),
                    "infinite": int(np.isinf(arr).sum()),
                    "n_classes": df["label"].nunique(),
                }
            )

    report_df = pd.DataFrame(reports)
    pd.set_option("display.width", 200)
    print("\n=== FILE CHECK (all 8 files) ===")
    print(report_df.to_string(index=False))

    # ---- Feature schema comparison ----
    print("\n=== FEATURE SCHEMA ===")
    all_sets = [set(v) for v in feature_sets.values()]
    common = [
        c
        for c in feature_sets[(15, "14class")]
        if all(c in s for s in all_sets)
    ]
    print(f"Common features across all 8 files: {len(common)}")
    print(common)

    for key, feats in feature_sets.items():
        extra = [c for c in feats if c not in common]
        print(f"{key[1]:8s} {key[0]:>3d}s  extra (not common): {extra}")

    # ---- Feature order identical to 15s? ----
    ref = feature_sets[(15, "14class")]
    print("\nSame feature list AND order as 15s Normal?")
    for key, feats in feature_sets.items():
        print(f"  {key[1]:8s} {key[0]:>3d}s -> {feats == ref}")

    # ---- Labels ----
    print("\n=== LABELS ===")
    for task in ["14class", "7class"]:
        for t in TIMEOUTS:
            df, _ = load_arff(t, task)
            print(
                f"{task} {t}s ({df['label'].nunique()} classes): {sorted(df['label'].unique())}"
            )


if __name__ == "__main__":
    check_all()