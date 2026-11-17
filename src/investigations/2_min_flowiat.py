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
    print(f"MIN_FLOWIAT INVESTIGATION: {file_path.name}")
    print("=" * 65)

    df = load_clean_arff(file_path)

    # Ensure min_flowiat is evaluated numerically
    df["min_flowiat"] = pd.to_numeric(df["min_flowiat"], errors="coerce")
    negative_flowiat = df[df["min_flowiat"] < 0]

    print("Total negative records:", len(negative_flowiat))

    print("\nDistribution of negative values:")
    print(negative_flowiat["min_flowiat"].value_counts().head(20))

    print("\nExamples:")
    print(negative_flowiat[["min_flowiat", "class1"]].head(20))