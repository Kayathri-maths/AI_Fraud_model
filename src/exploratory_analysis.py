import pandas as pd

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# 1. Check duplicate records
print("\nDuplicate Records: ", df.duplicated().sum())

# 2. Check transaction counts
print("\nTransaction Counts: ", df["Class"].value_counts())

# 3. Compare transaction amounts by class
print(
    "\nTransaction Amount Analysis: ",
    df.groupby("Class")["Amount"].agg(["count", "mean", "median", "min", "max"]),
)

# 4. Calculate fraud to legitimate ratio
legitimate = (df["Class"] == 0).sum()
fraud = (df["Class"] == 1).sum()

print("\nFraud to Legitimate ratio: ")
print(f"1 : {legitimate / fraud:.2f}")

# 5. Check transaction time range
print("\nTransaction time range: ")
print("Minimum: ", df["Time"].min())
print("Maximum: ", df["Time"].max())
