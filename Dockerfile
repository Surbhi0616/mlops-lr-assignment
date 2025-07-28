FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy required files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY artifacts/ artifacts/

# Run prediction
CMD ["python", "src/predict.py"]
