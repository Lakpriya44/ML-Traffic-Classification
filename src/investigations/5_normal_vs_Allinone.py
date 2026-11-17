from io import StringIO
from pathlib import Path
import pandas as pd
from scipy.io import arff

normal_path = Path("data/raw/Scenario_B/TimeBasedFeatures-Dataset-15s.arff")
allinone_path = Path(
    "data/raw/Scenario_B/Allinone/"
    "TimeBasedFeatures-Dataset-15s-AllinOne.arff"
)


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


print("\nNORMAL vs ALLINONE FEATURE COMPARISON")
print("=" * 70)

df_normal = load_clean_arff(normal_path)
df_allinone = load_clean_arff(allinone_path)

print(f"Normal shape:   {df_normal.shape}")
print(f"AllinOne shape: {df_allinone.shape}\n")

feature_columns = [
    column for column in df_normal.columns if column != "class1"
]

normal_features = df_normal[feature_columns].reset_index(drop=True)
allinone_features = df_allinone[feature_columns].reset_index(drop=True)

print("Identical feature matrices:", normal_features.equals(allinone_features))
print("Identical row order:", normal_features.equals(allinone_features))

print("\nTarget comparison:")
if len(df_normal) == len(df_allinone):
    print(
        pd.crosstab(
            df_normal["class1"].reset_index(drop=True),
            df_allinone["class1"].reset_index(drop=True),
            rownames=["Normal"],
            colnames=["AllinOne"],
        )
    )
else:
    print(
        "Row counts differ; showing separate class distributions instead:\n"
    )
    comparison = pd.DataFrame(
        {
            "Normal": df_normal["class1"].value_counts(),
            "AllinOne": df_allinone["class1"].value_counts(),
        }
    ).fillna(0).astype(int)
    print(comparison)