import pandas as pd
from pathlib import Path

# 1.Define dataset path
DATA_PATH = Path("data/creditcard.csv")

# 2. Check whether dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# 3. Load CSV dataset
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET LOADED ==========")

# 4.Display dataset dimentions
print("\nDataset shape: ", df.shape)

# 5. Display dataset 5 records
print("\nFirst 5 records: ", df.head())

# 6. Display column names
print("\nColumn names: ", df.columns.tolist())

# 7.Display data types and non-null counts
print("\nDataset Information: ", df.info())

# 8. Check missing values
print("\nMissing values: ", df.isnull().sum())

# 9. Statistical summary
print("\nStatistical summary: ", df.describe())

# 10. Check fraud and legitimate transactions
print("\nTransaction class distribution: ", df["Class"].value_counts())

# 11. Calculate transaction percentage
print("\nTransaction Percentage: ", df["Class"].value_counts(normalize=True) * 100)
