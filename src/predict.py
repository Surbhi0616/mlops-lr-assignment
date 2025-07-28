import joblib

def main():
    # Load model
    model = joblib.load("artifacts/linear_model.joblib")

    # Load test input
    X_test = joblib.load("artifacts/sample_input.joblib")

    # Predict
    preds = model.predict(X_test)

    # Print sample predictions
    print("Sample Predictions:", preds[:5])

if __name__ == "__main__":
    main()
