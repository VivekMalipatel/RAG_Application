FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to leverage Docker caching
COPY requirements.txt .

# Create required directories
RUN mkdir -p ollama/llama.cpp

RUN pip install --no-cache-dir packaging

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone llama.cpp repository inside the ollama directory
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ollama/llama.cpp

# Copy application code
COPY . .

# Set non-value environment variables (no defaults)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Expose port 
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]