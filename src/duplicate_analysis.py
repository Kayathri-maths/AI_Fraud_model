import pandas as pd

df = pd.read_csv("data/creditcard.csv")

# 1. Count exact duplicate rows
print("\nExact duplicate count: ", df.duplicated().sum())

# 2. Find all rows belonging to duplicate groups
duplicates = df[df.duplicated(keep=False)]

print("\nTotal rows involved in duplicate groups: ", len(duplicates))

# 3. Check duplicate records by transaction class
print("\nDuplicate groups by class: ")
print(duplicates["Class"].value_counts())

# 4. Check repeated records excluding the target column
feature_columns = df.columns.drop("Class")

feature_duplicates = df[df.duplicated(subset=feature_columns, keep=False)]

print("\nRepeated feature combinations (excluding Class): ", len(feature_duplicates))

# 5. Check whether repeated feature combinations have conflicating labels
label_counts = feature_duplicates.groupby(list(feature_columns))["Class"].nunique()

print("\nFeature combinations with conflicating labels: ", (label_counts > 1).sum())

# 6. display a few duplicate examples
print("\nSample duplicate records: ", duplicates.head(10))
