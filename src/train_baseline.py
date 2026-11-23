import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load cleaned dataset
df = pd.read_csv("data/creditcard_cleaned.csv")

# Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# Display information
print("Dataset Shape:", df.shape)
print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Distribution:")
print(y.value_counts())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Fraud Distribution:")
print(y_train.value_counts())

print("\nTesting Fraud Distribution:")
print(y_test.value_counts())


# Scale Time and Amount features
scaler = StandardScaler()

X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])

X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])


# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")


# Make predictions on test data
y_pred = model.predict(X_test)

# Predict probabilities
y_prob = model.predict_proba(X_test)

print("\nFirst 10 Predictions:")

for i in range(10):
    print(
        f"Actual: {y_test.iloc[i]} | Predicted: {y_pred[i]} | Fraud Probability: {y_prob[i][1]:.4f}"
    )
