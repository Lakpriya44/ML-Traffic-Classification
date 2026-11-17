from io import StringIO
from pathlib import Path
import pandas as pd
from scipy.io import arff

files = [
    Path("data/raw/Scenario_B/TimeBasedFeatures-Dataset-15s.arff"),
    Path(
        "data/raw/Scenario_B/Allinone/"
        "TimeBasedFeatures-Dataset-15s-AllinOne.arff"
    ),
]

TARGET = "class1"

for file_path in files:
    print("\n" + "=" * 65)
    print(f"DUPLICATE INVESTIGATION: {file_path.name}")
    print("=" * 65)

    lines = file_path.read_text(encoding="utf-8-sig").splitlines(keepends=True)

    # Locate the @DATA declaration
    data_index = next(
        i
        for i, line in enumerate(lines)
        if line.lstrip().lower().startswith("@data")
    )

    header = lines[: data_index + 1]
    data_lines = lines[data_index + 1 :]

    # Identify completely empty comma-only records
    blank_records = [
        (i + data_index + 2, line)
        for i, line in enumerate(data_lines)
        if line.strip() and set(line.strip()) <= {","}
    ]

    print("Completely blank records:", len(blank_records))
    for line_number, _ in blank_records:
        print("Blank record at physical line:", line_number)

    # Exclude blank records temporarily in memory
    valid_data_lines = [
        line
        for line in data_lines
        if not (line.strip() and set(line.strip()) <= {","})
    ]

    arff_content = "".join(header + valid_data_lines)
    data, _ = arff.loadarff(StringIO(arff_content))
    df = pd.DataFrame(data)

    print("\nDATASET OVERVIEW")
    print("Valid records:", len(df))
    print("Columns:", len(df.columns))
    print("Target classes:", df[TARGET].nunique())

    # Full-row duplicates
    full_duplicates = df.duplicated(keep=False)
    print("\nRows involved in full duplicates:", full_duplicates.sum())
    print("Extra duplicate occurrences:", df.duplicated().sum())

    # Duplicate feature vectors (ignoring target)
    X = df.drop(columns=[TARGET])
    feature_duplicates = X.duplicated(keep=False)
    print("\nRows with duplicate feature vectors:", feature_duplicates.sum())

    # Check whether identical feature vectors have conflicting labels
    duplicate_groups = df.groupby(list(X.columns), dropna=False)[TARGET].agg(
        ["size", "nunique"]
    )
    conflicting_groups = duplicate_groups[duplicate_groups["nunique"] > 1]

    print(
        "\nFeature groups with conflicting labels:", len(conflicting_groups)
    )
    print("Records in conflicting groups:", conflicting_groups["size"].sum())

    print("\nDuplicate full-row records by class:")
    print(df.loc[full_duplicates, TARGET].value_counts())