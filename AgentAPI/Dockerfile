# Use Python 3.11 slim image as base
FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port (if running as a web service)
EXPOSE 8000

# Default command to run the semantic store demo
CMD ["python", "pretests/semantic_store_langgraph.py"]
