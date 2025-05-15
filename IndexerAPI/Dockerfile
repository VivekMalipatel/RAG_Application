FROM python:3.11-slim

WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Expose the correct port
EXPOSE 8009

# Run the app on port 8010 to match EXPOSE
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8009"]