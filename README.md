MLOps Major Assignment: Linear Regression Pipeline

This repository implements a complete MLOps pipeline using a simple Linear Regression model on the California Housing dataset. The project focuses on automating the machine learning lifecycle using Python, Docker, and GitHub Actions.

---

## Project Features

- Uses `scikit-learn`'s `LinearRegression` model
- California Housing dataset from `sklearn.datasets`
- Manual quantization to uint8 format
- Dockerized prediction workflow
- CI/CD using GitHub Actions (on `main` branch push)

---

## Folder Structure

.
├── src/ # All source code
│ ├── train.py # Model training script
│ ├── quantize.py # Manual quantization script
│ ├── predict.py # Prediction script for Docker run
├── tests/
│ └── test_train.py # Unit tests for training pipeline
├── artifacts/ # Auto-generated models, inputs, weights
├── Dockerfile # Defines Docker container build
├── requirements.txt # Python dependencies
├── .github/workflows/ci.yml # CI/CD pipeline definition
├── .gitignore
└── README.md



---

## How to Run Locally

python src/train.py
python src/quantize.py
python src/predict.py
Sample Output:
bash
Copy code
R2 Score: 0.57
Mean Squared Error: 0.55
Quantized Inference Sample: [2.91 2.39 2.11 2.76 3.52]
Sample Predictions: [2.91 2.39 2.11 2.76 3.52]
How to Run with Docker
Make sure Docker is installed and running:

docker build -t mlops-lr .
docker run mlops-lr
Sample Docker Output:
bash
Copy code
R2 Score: 0.57
Mean Squared Error: 0.55
Sample Predictions: [2.91 2.39 2.11 2.76 3.52]
CI/CD Workflow (GitHub Actions)
A GitHub Actions workflow (ci.yml) automates the pipeline on every push to the main branch.

Jobs in CI/CD:
Stage	Description
Test Suite	Runs pytest on test_train.py to validate training steps
Train & Quantize	Executes train.py and quantize.py
Docker Build & Test	Builds Docker image and runs predict.py inside container

Workflow Summary:
Training model and saving .joblib outputs to artifacts/

Manually quantizing weights to uint8

Building and testing a Docker container for predictions

All steps run cleanly in CI/CD

Comparison Table (Required)
Step	Description
Training	LinearRegression trained on CA Housing dataset
Testing	Unit tests for data loading, model, R2 > 0.5
Quantization	Manual conversion to uint8 + dequantized inference
Dockerization	Container with predict script, auto-trains model
CI/CD	Full automation via GitHub Actions workflow

Notes & Constraints
Only LinearRegression was used as required

Python 3.10 or 3.9 (based on compatibility)

Modular code with utility separation

No hardcoded values used

All artifacts (*.joblib) are auto-generated and not committed

Only main branch used

Conclusion
This project demonstrates how to build, test, quantize, containerize, and deploy a simple ML model using modern DevOps principles. The entire lifecycle is automated and traceable using CI/CD pipelines.

