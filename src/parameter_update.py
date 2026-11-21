# Initial values
x = 2
actual = 25

weight = 2
bias = 0

learning_rate = 0.1
gradient = -5

# Initial prediction
prediction_before = (weight * x) + bias

# Calculate error
error_before = actual - prediction_before

# Update weight
new_weight = weight - (learning_rate * gradient)

# Prediction after updating
prediction_after = (new_weight * x) + bias

# Calculate new error
error_after = actual - prediction_after

print("Initial Weight:", weight)
print("Updated Weight:", new_weight)

print("\nBefore Update")
print("Prediction:", prediction_before)
print("Error:", error_before)

print("\nAfter Update")
print("Prediction:", prediction_after)
print("Error:", error_after)
