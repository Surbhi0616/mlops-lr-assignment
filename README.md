# MLOps Linear Regression Assignment

This project builds an end-to-end MLOps pipeline using:
- Scikit-learn's Linear Regression
- California Housing dataset
- Manual quantization
- Docker
- CI/CD with GitHub Actions

## Directory Structure

.
├── src/
│ ├── train.py
│ ├── quantize.py
│ ├── predict.py
│ └── utils.py
├── tests/
│ └── test_train.py
├── .github/
│ └── workflows/
│ └── ci.yml
├── artifacts/
├── Dockerfile
├── requirements.txt
├── .gitignore
├── README.md

vbnet
Copy code

## Steps Covered

| Step              | Description                         |
|-------------------|-------------------------------------|
| Training          | LinearRegression on CA dataset      |
| Testing           | PyTest: R2, model checks            |
| Quantization      | Manual to uint8, dequant inference  |
| Dockerization     | Container for prediction            |
| CI/CD             | GH Actions: test → train → docker   |
