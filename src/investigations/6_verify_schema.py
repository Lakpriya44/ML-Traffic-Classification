
from pathlib import Path
import pandas as pd
from scipy.io import arff

DATA_DIR = Path("data/raw/Scenario_B")

FILES = [
    DATA_DIR / "TimeBasedFeatures-Dataset-15s.arff",
    DATA_DIR / "Allinone" / "TimeBasedFeatures-Dataset-15s-AllinOne.arff",
]

for file_path in FILES:
    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    data, metadata = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # Remove completely blank records, if any
    df = df.dropna(how="all")

    print(f"Shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumn names:")
    for index, column in enumerate(df.columns, start=1):
        print(f"{index:2}. {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nTarget column:")
    print(df["class1"].value_counts().head())
