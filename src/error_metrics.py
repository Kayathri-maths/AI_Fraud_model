from sklearn.metrics import mean_absolute_error, mean_squared_error

# Actual values
y_actual = [100, 200, 300]

# Predicted values
y_predicted = [90, 210, 280]

# Calculate MAE
mae = mean_absolute_error(y_actual, y_predicted)

# Calculate MSE
mse = mean_squared_error(y_actual, y_predicted)

print("Actual Values:", y_actual)
print("Predicted Values:", y_predicted)

print("MAE:", mae)
print("MSE:", mse)
