import pandas as pd
from pathlib import Path

# File paths
INPUT_PATH = Path("data/creditcard.csv")
OUTPUT_PATH = Path("data/creditcard_cleaned.csv")


def load_dataset(file_path):
    """Load the CSV dataset."""

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    return df


def inspect_dataset(df):
    """Check dataset structure and data quality."""

    print("\nOriginal Dataset Shape:", df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    print("\nData Types:")
    print(df.dtypes.value_counts())


def clean_dataset(df):
    """Clean the dataset."""

    cleaned_df = df.copy()

    # Remove exact duplicate rows
    cleaned_df = cleaned_df.drop_duplicates(keep="first")

    # Check missing values
    if cleaned_df.isnull().sum().sum() > 0:
        raise ValueError("Missing values detected in dataset.")

    # Validate target column
    if "Class" not in cleaned_df.columns:
        raise ValueError("Target column 'Class' is missing.")

    # Validate target values
    if not cleaned_df["Class"].isin([0, 1]).all():
        raise ValueError("Unexpected values found in Class column.")

    print("\nCleaning completed.")

    return cleaned_df


def save_dataset(df, file_path):
    """Save the cleaned dataset."""

    file_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(file_path, index=False)

    print(f"\nCleaned dataset saved: {file_path}")


def main():

    # Load
    df = load_dataset(INPUT_PATH)

    # Inspect
    inspect_dataset(df)

    # Clean
    cleaned_df = clean_dataset(df)

    # Save
    save_dataset(cleaned_df, OUTPUT_PATH)

    print("\nOriginal Shape:", df.shape)
    print("Cleaned Shape:", cleaned_df.shape)

    print("\nFinal Class Distribution:")
    print(cleaned_df["Class"].value_counts())


if __name__ == "__main__":
    main()
