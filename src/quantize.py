import joblib
import numpy as np
import os

def quantize_weights(weights, scale=100):
    """Quantize to uint8"""
    return np.clip(weights * scale, 0, 255).astype(np.uint8)

def dequantize_weights(weights_q, scale=100):
    """Dequantize back to float"""
    return weights_q.astype(float) / scale

def main():
    os.makedirs("artifacts", exist_ok=True)

    # Load trained model
    model = joblib.load("artifacts/linear_model.joblib")

    # Extract weights
    coef = model.coef_
    intercept = model.intercept_

    # Save raw weights
    joblib.dump({"coef": coef, "intercept": intercept}, "artifacts/unquant_params.joblib")

    # Quantize
    coef_q = quantize_weights(coef)
    intercept_q = int(np.clip(intercept * 100, 0, 255))

    joblib.dump({"coef_q": coef_q, "intercept_q": intercept_q}, "artifacts/quant_params.joblib")

    # Dequantize
    coef_dq = dequantize_weights(coef_q)
    intercept_dq = intercept_q / 100

    # Inference
    X = joblib.load("artifacts/sample_input.joblib")
    pred = np.dot(X, coef_dq) + intercept_dq

    print("Quantized Inference Sample:", pred[:5])

    joblib.dump({"coef_q": coef_q, "intercept_q": intercept_q}, "artifacts/quant_params.joblib")
    print("Saved: artifacts/unquant_params.joblib")
    print("Saved: artifacts/quant_params.joblib")

    # Dequantize
    ...


if __name__ == "__main__":
    main()
