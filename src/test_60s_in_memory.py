
from pathlib import Path
from io import StringIO

import pandas as pd
from scipy.io import arff


files = [
    Path("data/raw/Scenario_B/TimeBasedFeatures-Dataset-60s.arff"),

    Path(
        "data/raw/Scenario_B/Allinone/"
        "TimeBasedFeatures-Dataset-60s-AllinOne.arff"
    )
]


for file_path in files:

    print("\n" + "=" * 65)
    print(file_path.name)
    print("=" * 65)

    lines = file_path.read_text(
        encoding="utf-8-sig"
    ).splitlines(keepends=True)

    # Locate the @DATA declaration
    data_index = next(
        i for i, line in enumerate(lines)
        if line.lstrip().lower().startswith("@data")
    )

    # Separate header and data
    header = lines[:data_index + 1]
    data_lines = lines[data_index + 1:]

    # Identify completely empty comma-only records
    blank_records = [
        (i + data_index + 2, line)
        for i, line in enumerate(data_lines)
        if line.strip()
        and set(line.strip()) <= {","}
    ]

    print("Completely blank records:", len(blank_records))

    for line_number, line in blank_records:
        print("Blank record at physical line:", line_number)

    # Exclude blank records temporarily in memory
    valid_data_lines = [
        line for line in data_lines
        if not (
            line.strip()
            and set(line.strip()) <= {","}
        )
    ]

    # Load without writing anything to the original file
    arff_content = "".join(header + valid_data_lines)

    data, metadata = arff.loadarff(
    StringIO(arff_content)
    )

    df = pd.DataFrame(data)

    print("Valid records:", len(df))
    print("Columns:", len(df.columns))
    print("Exact duplicate rows:", df.duplicated().sum())

    print("Target classes:", df["class1"].nunique())
