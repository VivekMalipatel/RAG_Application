# ModelRouter API

An OpenAI-compatible API server that routes requests to different model providers (OpenAI, HuggingFace, Ollama) based on the specified model.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API that can be used with existing libraries and applications
- **Multiple Model Providers**: Support for OpenAI, HuggingFace, and Ollama models
- **Standard Endpoints**: Implements standardized endpoints for chat completions, text completions, and embeddings
- **API Key Authentication**: Secure your API with API key authentication
- **Usage Tracking**: Log and track model usage for billing and analytics

## API Endpoints

The API follows the OpenAI API structure:

- `/v1/chat/completions`: Chat completion endpoint (like OpenAI's ChatGPT)
- `/v1/completions`: Text completion endpoint
- `/v1/embeddings`: Generate embeddings for text
- `/v1/models`: List available models

## Installation

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)

### Local Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ModelRouterAPI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables (or create a `.env` file):
   ```
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_API_TOKEN=your_huggingface_token
   OLLAMA_BASE_URL=http://localhost:11434
   ```

4. Run the server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Docker Installation

1. Build and run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

## Usage

### Authentication

Use the API key header in your requests:

```
X-Api-Key: your-api-key
```

Default API key for testing: `test-key`

### Example: Chat Completion

```python
import openai

client = openai.OpenAI(
    api_key="test-key",
    base_url="http://localhost:8000/v1"  # Point to your ModelRouter API
)

# Using OpenAI model
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about machine learning."}
    ]
)

# Using Ollama model
response = client.chat.completions.create(
    model="llama2",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about machine learning."}
    ]
)

print(response.choices[0].message.content)
```

### Example: Embeddings

```python
import openai

client = openai.OpenAI(
    api_key="test-key",
    base_url="http://localhost:8000/v1"
)

response = client.embeddings.create(
    model="text-embedding-ada-002",
    input="Machine learning is fascinating"
)

print(response.data[0].embedding)
```

## Configuration

The API can be configured through environment variables or the `.env` file:

- `API_KEYS`: Comma-separated list of valid API keys
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_BASE_URL`: Base URL for OpenAI API
- `HUGGINGFACE_API_TOKEN`: Your HuggingFace API token
- `OLLAMA_BASE_URL`: Base URL for Ollama API
- `DATABASE_URL`: Database connection string (defaults to SQLite)

## Database

The API uses SQLAlchemy with SQLite by default, but can be configured to use PostgreSQL by changing the `DATABASE_URL` environment variable:

```
# SQLite (default)
DATABASE_URL=sqlite:///./modelrouter.db

# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost/modelrouter
```