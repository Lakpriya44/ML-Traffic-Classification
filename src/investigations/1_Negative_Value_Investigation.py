from io import StringIO
from pathlib import Path
import numpy as np
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


def load_clean_arff(file_path: Path) -> pd.DataFrame:
    lines = file_path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    data_index = next(
        i
        for i, line in enumerate(lines)
        if line.lstrip().lower().startswith("@data")
    )
    header = lines[: data_index + 1]
    data_lines = lines[data_index + 1 :]

    # Exclude completely empty comma-only records
    valid_data_lines = [
        line
        for line in data_lines
        if not (line.strip() and set(line.strip()) <= {","})
    ]

    arff_content = "".join(header + valid_data_lines)
    data, _ = arff.loadarff(StringIO(arff_content))
    return pd.DataFrame(data)


for file_path in files:
    print("\n" + "=" * 65)
    print(f"NEGATIVE-VALUE INVESTIGATION: {file_path.name}")
    print("=" * 65)

    df = load_clean_arff(file_path)

    # Separate features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Convert predictors for inspection only
    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    conversion_errors = X.notna() & X_numeric.isna()

    print("\nMISSING VALUES")
    print(X_numeric.isna().sum())

    print("\nINFINITE VALUES")
    print(np.isinf(X_numeric).sum())

    print("\nCONVERSION ERRORS")
    print(conversion_errors.sum())

    print("\nNEGATIVE VALUES COUNT")
    print((X_numeric < 0).sum())

    print("\nZERO VALUES COUNT")
    print((X_numeric == 0).sum())

    print("\nSTATISTICAL SUMMARY")
    print(X_numeric.describe().T)

    print("\nDETAILED NEGATIVE VALUE BREAKDOWN")
    print("=" * 65)

    negative_features = X_numeric.columns[(X_numeric < 0).any()]

    for column in negative_features:
        print(f"\nFeature: {column}")
        negative_values = X_numeric.loc[X_numeric[column] < 0, column]

        print("Negative count:", len(negative_values))
        print("\nNegative value frequencies:")
        print(negative_values.value_counts().head(15))

        print("\nNegative values by target class:")
        print(pd.crosstab(y, X_numeric[column] < 0))