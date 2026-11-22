from sklearn.linear_model import LogisticRegression

# Sample input data
X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

# Create the model
model = LogisticRegression()

# Train the model
model.fit(X, y)

# Predict for new data
prediction = model.predict([[3.5]])

# Predict probability
probability = model.predict_proba([[3.5]])

print("Predicted Class:", prediction[0])
print("Probability:", probability[0])

# Test multiple inputs
test_values = [[1], [2], [3], [3.5], [4], [5], [6]]

for value in test_values:
    prediction = model.predict([value])
    probability = model.predict_proba([value])

    print(f"\nInput: {value[0]}")
    print(f"Predicted Class: {prediction[0]}")
    print(f"Class 0 Probability: {probability[0][0]:.4f}")
    print(f"Class 1 Probability: {probability[0][1]:.4f}")
