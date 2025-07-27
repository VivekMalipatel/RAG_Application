# servers/file-operations/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY ../../core/base-server/src/ ./core/base-server/src/

# Create data directory
RUN mkdir -p /app/data

# Expose the port
EXPOSE 8080

# Run the server
CMD ["python", "src/server.py"]
