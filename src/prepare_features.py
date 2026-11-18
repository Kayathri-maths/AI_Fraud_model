import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# define dataset path
DATA_PATH = Path("data/creditcard_cleaned.csv")

# load cleaned dataset
df = pd.read_csv(DATA_PATH)

# display dataset information
print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Information:")
print(df.info())

# seperate the input features and target variables
X = df.drop(columns=["Class"])
y = df["Class"]

# display the shapes
print("\nInput feature shape: ", X.shape)
print("Target variable shape: ", y.shape)

# diplay feature columns
print("\nInput feature columns: ", X.columns.tolist())
print("Target variable distribution: ", y.value_counts())


# split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# Display training and testing shapes
print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# Check class distribution after splitting
print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())


# inspect feature ranges in training data
print("\nTraining Features Statistics: ", X_train[["Time", "Amount"]].describe())
