# OmniRAG Application




## Setup

# docker activation
docker compose up -d

# docker deactivation
docker compose down

# docker status
docker ps

1. Create virtual environment:
```bash
python3 -m venv .rag
source .rag/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

