FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    ffmpeg \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /app
COPY . /app

# Ensure Tesseract is available
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV PATH="/usr/bin:${PATH}"

CMD ["python3", "app/main.py"]