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
    df = pd.DataFrame(data)

    # Decode byte strings in target column if present
    if "class1" in df.columns and df["class1"].dtype == object:
        df["class1"] = df["class1"].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )
    return df


columns_to_display = [
    "duration",
    "min_fiat",
    "min_biat",
    "max_fiat",
    "max_biat",
    "min_flowiat",
    "max_flowiat",
    "mean_flowiat",
    "std_flowiat",
    "class1",
]

for file_path in files:
    print("\n" + "=" * 70)
    print(f"UNUSUAL MIN_FLOWIAT RECORDS (< -1): {file_path.name}")
    print("=" * 70)

    df = load_clean_arff(file_path)
    df["min_flowiat"] = pd.to_numeric(df["min_flowiat"], errors="coerce")

    negative_flowiat = df[df["min_flowiat"] < -1]
    print("Number of records:", len(negative_flowiat))

    if not negative_flowiat.empty:
        print(
            negative_flowiat[columns_to_display]
            .sort_values("min_flowiat")
            .to_string(index=False)
        )