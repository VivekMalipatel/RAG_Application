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

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone llama.cpp repository inside the ollama directory
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ollama/llama.cpp

# Copy application code
COPY api/ /app/api/
COPY core/ /app/core/
COPY db/ /app/db/
COPY huggingface/ /app/huggingface/
COPY ollama/ /app/ollama/
COPY openai_client/ /app/openai_client/
COPY schemas/ /app/schemas/
COPY __init__.py /app/__init__.py
COPY config.py /app/config.py
COPY main.py /app/main.py
COPY model_handler.py /app/model_handler.py
COPY model_handler_v2.py /app/model_handler_v2.py
COPY model_type.py /app/model_type.py
COPY model_provider.py /app/model_provider.py

# Set non-value environment variables (no defaults)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV PORT=8080 
ENV WORKERS=1

# Command to run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}"]