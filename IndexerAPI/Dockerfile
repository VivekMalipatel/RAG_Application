FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    poppler-utils \
    libreoffice \
    python3-pip \
    python3-uno \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 -c "import sys; sys.path.append('/usr/lib/python3/dist-packages'); import uno; print('UNO library successfully imported')" || echo "UNO library not found, but continuing..."

COPY app/ ./
COPY supervisord.conf /etc/supervisord.conf

ENV PORT=8009
ENV WORKERS=1
ENV UNOSERVER_HOST=localhost
ENV UNOSERVER_PORT=2003
ENV PYTHONPATH="/app:/usr/lib/python3/dist-packages"
ENV UNO_PATH="/usr/lib/python3/dist-packages"
EXPOSE ${PORT}
EXPOSE 2003

CMD ["supervisord", "-c", "/etc/supervisord.conf"]