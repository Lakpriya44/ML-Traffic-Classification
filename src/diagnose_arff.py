
from pathlib import Path
import re

import pandas as pd
from scipy.io import arff


# --------------------------------------------------
# 1. Dataset locations
# --------------------------------------------------

DATA_DIR = Path("data/raw/Scenario_B")

# Find all ARFF files inside Scenario_B,
# including files inside the AllinOne subfolder.
ARFF_FILES = sorted(DATA_DIR.rglob("*.arff"))


# --------------------------------------------------
# 2. Read ARFF header and identify attributes
# --------------------------------------------------



def inspect_header(lines):

    attributes = []
    data_start = None

    for line_number, line in enumerate(lines, start=1):

        # Detect @ATTRIBUTE declarations
        if re.match(r"^\s*@attribute\b", line, re.IGNORECASE):

            match = re.match(
                r"""^\s*@attribute\s+(?:'[^']*'|"[^"]*"|\S+)\s+(.+?)\s*$""",
                line,
                re.IGNORECASE
            )

            if match:

                # Remove comma padding from the attribute type
                attribute_type = match.group(1).strip().rstrip(",")

                attributes.append(attribute_type)

        # Detect @DATA even when followed by commas
        if re.match(r"^\s*@data\b", line, re.IGNORECASE):

            data_start = line_number
            break

    return attributes, data_start




# --------------------------------------------------
# 3. Diagnose individual ARFF files
# --------------------------------------------------

def diagnose_file(file_path):

    print("\n" + "=" * 75)
    print(f"FILE: {file_path}")
    print("=" * 75)

    with open(file_path, "r", encoding="utf-8-sig") as file:
        lines = file.readlines()

    attributes, data_start = inspect_header(lines)

    if data_start is None:

        print("ERROR: Could not locate @DATA section.")
        return

    print(f"Number of attributes: {len(attributes)}")
    print(f"@DATA found at physical line: {data_start}")

    # Identify numerical columns
    numeric_columns = []

    for index, attribute_type in enumerate(attributes):

        if attribute_type.lower() in ("numeric", "real", "integer"):

            numeric_columns.append(index)

    print(f"Numerical attributes: {len(numeric_columns)}")

    # --------------------------------------------------
    # 4. Inspect actual data records
    # --------------------------------------------------

    total_records = 0
    blank_lines = 0
    comment_lines = 0

    invalid_records = []
    missing_values = 0

    for line_number in range(data_start, len(lines)):

        raw_line = lines[line_number]
        stripped = raw_line.strip()

        # Ignore empty lines
        if not stripped:
            blank_lines += 1
            continue

        # Ignore comments
        if stripped.startswith("%"):
            comment_lines += 1
            continue

        # This is a data record
        total_records += 1

        # These datasets contain numerical flow features
        # and a nominal target label.
        values = stripped.split(",")

        # Check the number of fields
        if len(values) != len(attributes):

            invalid_records.append({
                "line": line_number + 1,
                "record": total_records,
                "problem": "Incorrect number of columns",
                "details": (
                    f"Expected {len(attributes)}, "
                    f"found {len(values)}"
                )
            })

            continue

        # Inspect numerical values
        for column_index in numeric_columns:

            value = values[column_index].strip().strip("'\"")

            # ARFF missing-value marker
            if value == "?":
                missing_values += 1
                continue

            # Empty numerical value
            if value == "":
                invalid_records.append({
                    "line": line_number + 1,
                    "record": total_records,
                    "problem": "Empty numerical value",
                    "column": column_index + 1,
                    "attribute_type": attributes[column_index],
                    "details": f"Value: {repr(value)}"
                })

                continue

            # Check whether value can be converted to float
            try:
                float(value)

            except ValueError:

                invalid_records.append({
                    "line": line_number + 1,
                    "record": total_records,
                    "problem": "Invalid numerical value",
                    "column": column_index + 1,
                    "attribute_type": attributes[column_index],
                    "details": f"Value: {repr(value)}"
                })

    # --------------------------------------------------
    # 5. Print diagnostic results
    # --------------------------------------------------

    print("\nDATA INSPECTION SUMMARY")

    print(f"Total data records: {total_records}")
    print(f"Blank lines: {blank_lines}")
    print(f"Comment lines: {comment_lines}")
    print(f"Missing numerical values (?): {missing_values}")

    print(f"Invalid records detected: {len(invalid_records)}")

    if invalid_records:

        print("\nPROBLEMATIC RECORDS")

        for problem in invalid_records[:20]:

            print("\n------------------------------")

            for key, value in problem.items():
                print(f"{key}: {value}")

            # Print the original record without changing it
            print("Original line:")
            print(lines[problem["line"] - 1].rstrip())

        if len(invalid_records) > 20:
            print("\nOnly the first 20 errors are displayed.")

    else:

        print("\nNo obvious numerical or column-count errors found.")

    # --------------------------------------------------
    # 6. Test SciPy ARFF loading
    # --------------------------------------------------

    print("\nSCIPY ARFF LOADING TEST")

    try:

        data, metadata = arff.loadarff(file_path)

        dataframe = pd.DataFrame(data)

        print("Loading successful!")

        print(f"Rows: {len(dataframe)}")
        print(f"Columns: {len(dataframe.columns)}")

        print(
            f"Missing values: "
            f"{dataframe.isna().sum().sum()}"
        )

        print(
            f"Exact duplicate rows: "
            f"{dataframe.duplicated().sum()}"
        )

    except Exception as error:

        print("Loading failed!")

        print(f"Error type: {type(error).__name__}")
        print(f"Error message: {error}")


# --------------------------------------------------
# 7. Run inspection for every ARFF file
# --------------------------------------------------

print(f"Total ARFF files found: {len(ARFF_FILES)}")

for file_path in ARFF_FILES:

    diagnose_file(file_path)
