# Holmes AI - Transaction Categorization Engine
FROM python:3.10-slim

LABEL maintainer="Holmes AI Team"
LABEL description="AI-native transaction categorization with enterprise-grade accuracy"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY setup.py .

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p data/raw data/processed models logs

# Expose API port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command: start API server
CMD ["uvicorn", "holmes.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
