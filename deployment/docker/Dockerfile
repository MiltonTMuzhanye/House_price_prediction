FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p artifacts data/processed logs

# Expose ports
EXPOSE 8000 8501

# Start API
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]