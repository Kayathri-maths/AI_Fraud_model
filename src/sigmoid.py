import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Test inputs
values = [-5, -2, 0, 2, 5, 10, 99, 100]

print("Sigmoid Results")

for x in values:
    result = sigmoid(x)
    print(f"Input: {x} -> Sigmoid: {result:.4f}")
