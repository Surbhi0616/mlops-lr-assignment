from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def main():
    # Load dataset
    X, y = fetch_california_housing(return_X_y=True)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"R2 Score: {r2}")
    print(f"Mean Squared Error: {mse}")

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/linear_model.joblib")
    joblib.dump(X_test, "artifacts/sample_input.joblib")

if __name__ == "__main__":
    main()
