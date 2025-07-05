# Agent API

A sophisticated RAG (Retrieval-Augmented Generation) application built with FastAPI and LangGraph, providing powerful AI agents with chain-of-thought reasoning and deep research capabilities.

## Features

- **Single Shot Agent**: Direct, comprehensive responses to queries
- **Deep Research Agent**: Conducts comprehensive research with background gathering, intent analysis, and synthesis
- **RESTful API**: Full CRUD operations for agent management
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Memory Management**: Thread-based conversation persistence
- **Extensible Architecture**: Easy to add new agent types and capabilities

## Quick Start with Docker

### Prerequisites

- Docker
- Docker Compose

### 1. Clone and Setup

```bash
git clone <your-repo>
cd AgentAPI
```

### 2. Environment Configuration

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# API Keys
OPENAI_API_KEY=your-openai-key
OPENAI_API_BASE=http://your-model-router:8000/v1
TAVILY_API_KEY=your-tavily-key

# Application
ENVIRONMENT=production
LOG_LEVEL=info

# Model Router (your custom LLM endpoint)
MODEL_ROUTER_BASE_URL=http://192.5.87.119:8000/v1
MODEL_ROUTER_API_KEY=test-key
```

### 3. Start the Application

For production:
```bash
docker-compose up -d
```

For development (with hot reload):
```bash
docker-compose -f docker-compose.dev.yml up -d
```

### 4. Access the API

- **API Base URL**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Usage

### Create an Agent

```bash
curl -X POST "http://localhost:8000/api/v1/agents/create" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Research Assistant",
    "agent_type": "deep_research",
    "role": "Senior Research Analyst",
    "instructions": "You are an expert researcher who provides comprehensive analysis.",
    "capabilities": ["web_search", "analysis", "synthesis"]
  }'
```

### Execute an Agent

```bash
curl -X POST "http://localhost:8000/api/v1/agents/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "single_shot",
    "query": "Explain the impact of AI on modern software development",
    "thread_id": "my-conversation-1"
  }'
```

### Chat with a Specific Agent

```bash
curl -X POST "http://localhost:8000/api/v1/agents/{agent_id}/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in LangGraph?",
    "thread_id": "chat-session-1"
  }'
```

### List All Agents

```bash
curl "http://localhost:8000/api/v1/agents/list"
```

## Agent Types

### Single Shot Agent

The Single Shot agent provides direct, comprehensive responses to queries without complex reasoning steps:

**Characteristics**:
- Direct response to user queries
- Comprehensive and accurate answers
- Fast execution
- Suitable for straightforward questions

**Use Cases**:
- Quick answers
- Direct information retrieval
- Simple explanations
- FAQ responses

### Deep Research Agent

The Deep Research agent conducts comprehensive research:

1. **Background Knowledge**: Gathers foundational information
2. **Intent Analysis**: Understands research objectives and sub-questions
3. **Gap Assessment**: Identifies knowledge gaps and missing information
4. **Synthesis**: Creates a comprehensive final response

**Use Cases**:
- Market research
- Academic research
- Competitive analysis
- Technical documentation

## Docker Commands

### Production Deployment

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Update and restart
docker-compose pull && docker-compose up -d
```

### Development

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# View logs with hot reload
docker-compose -f docker-compose.dev.yml logs -f

# Stop development environment
docker-compose -f docker-compose.dev.yml down
```

### Useful Docker Commands

```bash
# Rebuild containers
docker-compose build --no-cache

# Access container shell
docker exec -it agent-api bash

# View container logs
docker logs agent-api

# Check container status
docker-compose ps
```

## Architecture

```
AgentAPI/
├── app/
│   ├── agents/
│   │   ├── base_types/
│   │   │   ├── single_shot.py
│   │   │   └── deep_research.py
│   │   └── base_agent.py
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── agents.py
│   │   │   └── health.py
│   │   └── routes.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   └── main.py
├── nginx/
│   └── nginx.conf
├── scripts/
├── docker-compose.yml
├── docker-compose.dev.yml
├── Dockerfile
└── requirements.txt
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key or custom API key | `test-key` |
| `OPENAI_API_BASE` | Base URL for LLM API | `http://192.5.87.119:8000/v1` |
| `TAVILY_API_KEY` | Tavily search API key | `your-tavily-key` |
| `ENVIRONMENT` | Environment (development/production) | `production` |
| `LOG_LEVEL` | Logging level | `info` |
| `API_PREFIX` | API route prefix | `/api/v1` |

## Health Monitoring

Check application health:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/api/v1/health/detailed
```

## Logging

Logs are available in the `logs/` directory and through Docker:

```bash
# View application logs
docker logs agent-api

# Follow logs in real-time
docker logs -f agent-api

# View all service logs
docker-compose logs -f
```

## Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   # Kill the process or change the port in docker-compose.yml
   ```

2. **Permission denied errors**:
   ```bash
   # Fix permissions for logs directory
   sudo chown -R $USER:$USER logs/
   ```

3. **Container fails to start**:
   ```bash
   # Check container logs
   docker logs agent-api
   # Rebuild without cache
   docker-compose build --no-cache
   ```

### Development Tips

- Use the development docker-compose file for hot reload
- Check the `/docs` endpoint for interactive API documentation
- Monitor logs for debugging: `docker-compose logs -f`
- Use the health endpoints to verify service status

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Docker
5. Submit a pull request

## License

[Your License Here]
