# Fraud probabilities predicted by our model
fraud_probabilities = [0.10, 0.25, 0.35, 0.40, 0.55, 0.80, 0.95]

# Threshold values we want to test
thresholds = [0.50, 0.30, 0.20]

# Analyze each threshold
for threshold in thresholds:
    print(f"\nThreshold: {threshold}")
    print("-" * 40)

    for probability in fraud_probabilities:
        if probability >= threshold:
            prediction = 1  # Fraud
        else:
            prediction = 0  # Legitimate

        print(f"Probability: {probability:.2f} -> Prediction: {prediction}")
