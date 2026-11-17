
from pathlib import Path
from scipy.io import arff
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENARIO_B_DIR = PROJECT_ROOT / "data" / "raw" / "Scenario_B"
ALL_IN_ONE_DIR = SCENARIO_B_DIR / "AllinOne"


def load_arff(file_path):
    data, metadata = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # Decode byte values in categorical columns
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].apply(
            lambda value: value.decode("utf-8")
            if isinstance(value, bytes)
            else value
        )

    return df, metadata


datasets = {
    "15s": SCENARIO_B_DIR / "TimeBasedFeatures-Dataset-15s.arff",
    "30s": SCENARIO_B_DIR / "TimeBasedFeatures-Dataset-30s.arff",
    "60s": SCENARIO_B_DIR / "TimeBasedFeatures-Dataset-60s.arff",
    "120s": SCENARIO_B_DIR / "TimeBasedFeatures-Dataset-120s.arff",

    "15s-AllinOne": ALL_IN_ONE_DIR / "TimeBasedFeatures-Dataset-15s-AllinOne.arff",
    "30s-AllinOne": ALL_IN_ONE_DIR / "TimeBasedFeatures-Dataset-30s-AllinOne.arff",
    "60s-AllinOne": ALL_IN_ONE_DIR / "TimeBasedFeatures-Dataset-60s-AllinOne.arff",
    "120s-AllinOne": ALL_IN_ONE_DIR / "TimeBasedFeatures-Dataset-120s-AllinOne.arff",
}


for dataset_name, file_path in datasets.items():

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    if not file_path.exists():
        print(f"FILE NOT FOUND: {file_path}")
        continue

    df, metadata = load_arff(file_path)

    print("\nDataset shape:", df.shape)
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 records:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate records:")
    print(df.duplicated().sum())

    print("\nARFF attributes:")
    for attribute in metadata.names():
        print(attribute, ":", metadata[attribute][0])
