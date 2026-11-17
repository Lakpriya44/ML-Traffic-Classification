
from pathlib import Path
from io import StringIO

import pandas as pd
from scipy.io import arff


DATA_DIR = Path("data/raw/Scenario_B")

files = sorted(DATA_DIR.rglob("*.arff"))


for file_path in files:

    print("\n" + "=" * 70)
    print(file_path.name)
    print("=" * 70)

    lines = file_path.read_text(
        encoding="utf-8-sig"
    ).splitlines(keepends=True)

    # Locate the @DATA declaration
    data_index = next(
        i for i, line in enumerate(lines)
        if line.lstrip().lower().startswith("@data")
    )

    header = lines[:data_index + 1]
    data_lines = lines[data_index + 1:]

    # Exclude completely blank comma-only records in memory
    valid_data_lines = [
        line for line in data_lines
        if not (
            line.strip()
            and set(line.strip()) <= {","}
        )
    ]

    arff_content = "".join(header + valid_data_lines)

    data, metadata = arff.loadarff(
        StringIO(arff_content)
    )

    df = pd.DataFrame(data)

    # Decode ARFF byte labels into readable strings
    df["class1"] = df["class1"].apply(
        lambda value: value.decode("utf-8")
        if isinstance(value, bytes)
        else value
    )

    # Calculate class distribution
    distribution = df["class1"].value_counts().reset_index()

    distribution.columns = [
        "Class",
        "Record Count"
    ]

    distribution["Percentage"] = (
        distribution["Record Count"]
        / len(df)
        * 100
    )

    print("Total records:", len(df))
    print("Number of classes:", df["class1"].nunique())

    print("\nClass Distribution:")
    print(
        distribution.to_string(
            index=False,
            formatters={
                "Percentage": "{:.2f}%".format
            }
        )
    )
