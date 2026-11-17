from io import StringIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.io import arff


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "Scenario_B"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

TIMEOUTS = [15, 30, 60, 120]

DATASET_FILES = {}

for timeout in TIMEOUTS:
    DATASET_FILES[f"{timeout}s_Normal"] = (
        RAW_DIR / f"TimeBasedFeatures-Dataset-{timeout}s.arff"
    )

    DATASET_FILES[f"{timeout}s_AllinOne"] = (
        RAW_DIR
        / "Allinone"
        / f"TimeBasedFeatures-Dataset-{timeout}s-AllinOne.arff"
    )


# ============================================================
# 3. VERIFIED FEATURE SCHEMA
# ============================================================

FEATURE_COLUMNS = [
    "duration",
    "total_fiat",
    "total_biat",
    "min_fiat",
    "min_biat",
    "max_fiat",
    "max_biat",
    "mean_fiat",
    "mean_biat",
    "flowPktsPerSecond",
    "flowBytesPerSecond",
    "min_flowiat",
    "max_flowiat",
    "mean_flowiat",
    "std_flowiat",
    "min_active",
    "mean_active",
    "max_active",
    "std_active",
    "min_idle",
    "mean_idle",
    "max_idle",
    "std_idle",
]

TARGET_COLUMN = "class1"

EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


# ============================================================
# 4. ARFF LOADING
# ============================================================


def load_arff_dataset(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # Read original file as text without modifying it.
    # encoding="utf-8-sig" automatically handles any UTF-8 BOM.
    raw_text = file_path.read_text(encoding="utf-8-sig")
    lines = raw_text.splitlines()

    cleaned_lines = []
    in_data_section = False
    excluded_blank_records = 0
    expected_col_count = len(EXPECTED_COLUMNS)

    for line in lines:
        stripped = line.strip()

        # Match @data even if followed by inline comments or whitespace
        if stripped.lower().startswith("@data"):
            in_data_section = True
            cleaned_lines.append("@data")
            continue

        if in_data_section:
            # Skip completely empty lines, comments, or comma-only records
            if (
                not stripped
                or stripped.startswith("%")
                or re.fullmatch(r",+", stripped)
            ):
                excluded_blank_records += 1
                continue

            # Split row and replace empty fields with ARFF missing marker "?"
            fields = [
                field.strip() if field.strip() != "" else "?"
                for field in stripped.split(",")
            ]

            # If a row is truncated (fewer than expected columns), pad with "?"
            if len(fields) < expected_col_count:
                fields.extend(["?"] * (expected_col_count - len(fields)))

            cleaned_lines.append(",".join(fields))
            continue

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    # Load the in-memory ARFF content via StringIO.
    records, metadata = arff.loadarff(StringIO(cleaned_text))

    df = pd.DataFrame(records)

    # Decode byte-string target labels (turning '?' into actual NaN if any existed).
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(
        lambda value: (
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
        ).strip()
    )
    df[TARGET_COLUMN] = df[TARGET_COLUMN].replace("?", np.nan)

    return df, excluded_blank_records


# ============================================================
# 5. DATASET AUDIT
# ============================================================


def audit_dataset(dataset_name, file_path):
    print("\n" + "=" * 75)
    print(f"DATASET: {dataset_name}")
    print("=" * 75)

    df, excluded_blank_records = load_arff_dataset(file_path)

    # --------------------------------------------------------
    # Schema validation (log differences instead of aborting)
    # --------------------------------------------------------

    actual_columns = list(df.columns)
    schema_matches = actual_columns == EXPECTED_COLUMNS

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(actual_columns))
    unexpected_columns = sorted(set(actual_columns) - set(EXPECTED_COLUMNS))

    print(f"\nShape: {df.shape}")
    print(f"Expected columns: {len(EXPECTED_COLUMNS)}")
    print(f"Actual columns: {len(actual_columns)}")
    print(f"Schema matches: {schema_matches}")
    print(f"Excluded blank records: {excluded_blank_records}")

    if not schema_matches:
        print("WARNING - Schema mismatch detected:")
        print("  Missing columns:", missing_columns)
        print("  Unexpected columns:", unexpected_columns)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing in {dataset_name}")

    # Use actual feature columns present in this file so audit can continue
    actual_feature_columns = [
        col for col in actual_columns if col != TARGET_COLUMN
    ]

    # --------------------------------------------------------
    # Numeric conversion validation
    # --------------------------------------------------------

    numeric_df = df[actual_feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    conversion_errors = (
        (numeric_df.isna() & df[actual_feature_columns].notna()).sum().sum()
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = int(df.isna().sum().sum())
    missing_feature_values = int(numeric_df.isna().sum().sum())

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    infinite_values = int(np.isinf(numeric_df.to_numpy(dtype=float)).sum())

    # --------------------------------------------------------
    # Negative values
    # --------------------------------------------------------

    negative_counts = (numeric_df < 0).sum()

    negative_features = {
        feature: int(count)
        for feature, count in negative_counts.items()
        if count > 0
    }

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_distribution = df[TARGET_COLUMN].value_counts().sort_index()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nDATA QUALITY RESULTS")
    print(f"Missing values: {missing_values}")
    print(f"Missing numeric values: {missing_feature_values}")
    print(f"Infinite values: {infinite_values}")
    print(f"Numeric conversion errors: {conversion_errors}")

    print("\nNegative values by feature:")
    if negative_features:
        for feature, count in negative_features.items():
            print(f"  {feature}: {count}")
    else:
        print("  None")

    print("\nCLASS DISTRIBUTION")
    print(class_distribution.to_string())

    print("\nTARGET INFORMATION")
    print("Unique classes:", df[TARGET_COLUMN].nunique())
    print("Target data type:", df[TARGET_COLUMN].dtype)

    # --------------------------------------------------------
    # Return audit results
    # --------------------------------------------------------

    summary = {
        "dataset": dataset_name,
        "records": len(df),
        "columns": len(df.columns),
        "schema_matches": schema_matches,
        "missing_columns": str(missing_columns) if missing_columns else "None",
        "unexpected_columns": (
            str(unexpected_columns) if unexpected_columns else "None"
        ),
        "excluded_blank_records": excluded_blank_records,
        "missing_values": missing_values,
        "missing_numeric_values": missing_feature_values,
        "infinite_values": infinite_values,
        "conversion_errors": int(conversion_errors),
        "unique_classes": df[TARGET_COLUMN].nunique(),
        "minimum_class_count": int(class_distribution.min()),
        "maximum_class_count": int(class_distribution.max()),
        "negative_features": str(negative_features),
    }

    distribution = class_distribution.rename_axis("class").reset_index(
        name="count"
    )
    distribution.insert(0, "dataset", dataset_name)

    return summary, distribution


# ============================================================
# 6. MAIN EXECUTION
# ============================================================


def main():
    all_summaries = []
    all_distributions = []

    print("\nISCX-VPN-NONVPN-2016 DATASET AUDIT")
    print("=" * 75)

    for dataset_name, file_path in DATASET_FILES.items():
        summary, distribution = audit_dataset(dataset_name, file_path)
        all_summaries.append(summary)
        all_distributions.append(distribution)

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    summary_df = pd.DataFrame(all_summaries)
    distribution_df = pd.concat(all_distributions, ignore_index=True)

    summary_path = REPORT_DIR / "dataset_audit_summary.csv"
    distribution_path = REPORT_DIR / "class_distributions.csv"

    summary_df.to_csv(summary_path, index=False)
    distribution_df.to_csv(distribution_path, index=False)

    print("\n" + "=" * 75)
    print("FINAL AUDIT SUMMARY")
    print("=" * 75)

    print(
        summary_df[
            [
                "dataset",
                "records",
                "schema_matches",
                "missing_values",
                "infinite_values",
                "conversion_errors",
                "unique_classes",
            ]
        ].to_string(index=False)
    )

    print("\nREPORTS GENERATED SUCCESSFULLY")
    print(f"1. {summary_path}")
    print(f"2. {distribution_path}")
    print("\nDATASET AUDIT COMPLETED.")


if __name__ == "__main__":
    main()