# API Directory

This directory contains the FastAPI applications for the Prompt Engineering Playground.

## Available APIs

### 1. Main Chat API (`main.py`)

The primary LLM chat API with streaming support.

**Port:** 8000 (default)

**Key Endpoints:**
- `GET /` - Root health check
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /chat` - Non-streaming chat
- `POST /chat.stream` - Streaming chat (raw text)
- `POST /chat.streamsse` - Streaming chat (SSE)

**Run:**
```bash
uvicorn api.main:app --reload --port 8000
```

**Features:**
- Multiple LLM providers (OpenAI, Google, Anthropic)
- Streaming and non-streaming responses
- MLflow tracing and logging
- OpenTelemetry integration
- Request/response logging to database
- Error handling and validation

---

### 2. Prompts Management API (`prompts.py`)

RESTful API for managing versioned prompt templates.

**Port:** 8001 (default)

**Key Endpoints:**
- `GET /prompts` - List latest prompts
- `POST /prompts` - Create new prompt
- `GET /prompts/{id}` - Get specific prompt
- `POST /prompts/{id}/versions` - Create new version
- `GET /prompts/{technique}/{title}/latest` - Get latest by name
- `POST /prompts/{id}/render` - Render with Jinja2
- `PATCH /prompts/{id}` - Update status/enabled

**Run:**
```bash
uvicorn api.prompts:app --reload --port 8001
# Or: ./run_prompts_api.sh
```

**Features:**
- Versioned prompt templates
- Jinja2 template rendering
- Search and filtering
- Lifecycle management (enable/disable/archive)
- Variable validation
- Complete CRUD operations

**Documentation:**
- Full API docs: `docs/PROMPTS_API.md`
- Setup guide: `PROMPTS_API_SETUP.md`
- Summary: `PROMPTS_API_SUMMARY.md`

---

## Running Both APIs

You can run both APIs simultaneously on different ports:

```bash
# Terminal 1: Main Chat API
uvicorn api.main:app --reload --port 8000

# Terminal 2: Prompts API
uvicorn api.prompts:app --reload --port 8001
```

Access:
- Chat API: http://localhost:8000/docs
- Prompts API: http://localhost:8001/docs

---

## Future Integration

These APIs can be integrated into a single application:

```python
from fastapi import FastAPI
from api.main import app as main_app
from api.prompts import app as prompts_app

app = FastAPI()
app.mount("/chat", main_app)
app.mount("/prompts", prompts_app)
```

Or use an API gateway/reverse proxy like nginx or Traefik.

---

## Architecture

Both APIs share:
- PostgreSQL database
- Environment variables (`.env`)
- Logging configuration
- Error handling patterns
- Pydantic validation

Separate concerns:
- Main API: Runtime LLM interactions
- Prompts API: Template management and versioning

