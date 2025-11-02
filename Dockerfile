# RAG Evaluation System - Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY evaluator.py .
COPY config.py .
COPY utils.py .
COPY example_usage.py .
COPY test_evaluator.py .

# Create results directory
RUN mkdir -p results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PHOENIX_PORT=6006

# Expose Phoenix port
EXPOSE 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:6006/ || exit 1

# Default command
CMD ["python", "evaluator.py"]