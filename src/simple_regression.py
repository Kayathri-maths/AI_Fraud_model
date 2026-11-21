from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([20, 35, 50, 65, 80])

print("Input Features (X):")
print(X)

print("Target Values (y):")
print(y)

model = LinearRegression()

model.fit(X, y)

print("Model Training Completed")
print("Weight:", model.coef_)
print("Bias:", model.intercept_)


new_hours = np.array([[6], [7], [8]])

predictions = model.predict(new_hours)

print("Predictions for new study hours:")
print(predictions)

print("\nResult Analysis")

print("Learned Equation: y =", model.coef_[0], "x +", model.intercept_)

for hours, marks in zip(new_hours.flatten(), predictions):
    print(f"{hours} study hours -> Predicted marks: {marks}")
