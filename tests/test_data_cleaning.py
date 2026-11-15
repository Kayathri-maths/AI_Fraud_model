import pandas as pd
import pytest

from src.data_cleaning import clean_dataset, load_dataset


def test_remove_duplicates():
    # Arrange: Create sample data
    df = pd.DataFrame({"Amount": [100, 200, 100], "Class": [0, 1, 0]})

    # Act: Clean the data
    cleaned_df = clean_dataset(df)

    # Assert: Verify duplicates removed
    assert len(cleaned_df) == 2
    assert cleaned_df.duplicated().sum() == 0


def test_preserve_original_dataset():
    df = pd.DataFrame({"Amount": [100, 200, 100], "Class": [0, 1, 0]})

    original_shape = df.shape

    cleaned_df = clean_dataset(df)

    assert df.shape == original_shape
    assert len(df) == 3


def test_invalid_class_values():
    df = pd.DataFrame({"Amount": [100, 200], "Class": [0, 2]})

    with pytest.raises(ValueError):
        clean_dataset(df)


def test_missing_values():
    df = pd.DataFrame({"Amount": [100, None], "Class": [0, 1]})

    with pytest.raises(ValueError):
        clean_dataset(df)


def test_missing_target_column():
    df = pd.DataFrame({"Amount": [100, 200]})

    with pytest.raises(ValueError):
        clean_dataset(df)


def test_missing_file():
    file_path = pytest.Path("non_existing.csv") if False else None

    with pytest.raises(FileNotFoundError):
        load_dataset(__import__("pathlib").Path("data/non_existing.csv"))
