import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# initialize standardscaler
scaler = StandardScaler()

# create copies of training and testing data
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# fit scaler using training data and transforms it
X_train_scaled[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])

# transform testing data using the same scaler
X_test_scaled[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

# display scaled feature statistics
print("\nScaled Training Feature Statistics:")
print(X_train_scaled[["Time", "Amount"]].describe())

# validate prpared features
print("\nFeature preparation validation: ")

print("Training data contains missing values:", X_train_scaled.isnull().sum().sum())
print("Testing data contains missing values:", X_test_scaled.isnull().sum().sum())

print("Training data shape:", X_train_scaled.shape)
print("Testing data shape:", X_test_scaled.shape)

print("\nScaled Training Mean:", X_train_scaled[["Time", "Amount"]].mean())
print(
    "\nScaled Training Standard Deviation: ", X_train_scaled[["Time", "Amount"]].std()
)
