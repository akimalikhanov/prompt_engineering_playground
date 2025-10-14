from fastapi import FastAPI, HTTPException, Query, Response, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Generator
from utils.errors import BackendError
from utils.sse import sse_pack
from config.load_models_config import _load_models_config
from schemas.schemas import *
from services.router import route_call
from utils.logging import setup_logging, correlation_id_middleware
from dotenv import load_dotenv
import os
import mlflow

load_dotenv()

# Ensure MLflow traces are exported both to MLflow Tracking and via OTLP to the collector
os.environ.setdefault("MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT", "true")

# Standardize OTLP settings for libraries that read either generic or traces-specific vars
os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")

# Send OTLP to local collector (exposed by docker-compose on host port 4318)
os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces")

# Service identity
os.environ.setdefault("OTEL_RESOURCE_ATTRIBUTES", "service.name=pep-api,service.namespace=pep,deployment.environment=prod")
os.environ.setdefault("OTEL_SERVICE_NAME", "pep-api")

# Reduce MLflow trace verbosity to prevent oversized traces
os.environ.setdefault("MLFLOW_TRACE_LOG_INPUTS_OUTPUTS", "false")
os.environ.setdefault("MLFLOW_TRACE_LOG_MODEL_INFERENCES", "false")

os.environ.setdefault('MLFLOW_LOG_LEVEL', 'INFO')

mlflow.set_tracking_uri(f"http://localhost:{os.getenv('MLFLOW_PORT')}")
mlflow.set_experiment("pep-playground")
mlflow.openai.autolog()


_models_config = _load_models_config()
_models_list = [model['label'] for model in _models_config['models']]

app = FastAPI(title="Prompt Engineering Playground")

logger = setup_logging(
    app_logger_name=os.getenv("LOGGER_NAME", "llm-router"),
    level=20,                 # logging.INFO
    to_console=True,
    to_file=True,
    file_path="logs/app.log", # stored locally in ./logs
    hijack_uvicorn=True,      # <- key to avoid duplicate uvicorn+app logs
)

app.middleware("http")(correlation_id_middleware)

# -------------------------
# Endpoints
# -------------------------
@app.get("/", response_model=RootResponse)
def read_root() -> RootResponse:
    return RootResponse(status="ok", message="Hello from FastAPI")

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")

@app.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(models=_models_list)

@mlflow.trace()
@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={200: {"description": "Complete response as JSON"}},
)
def chat(body: ChatRequest):
    result = route_call(
        provider_id=body.provider_id,
        model_id=body.model_id,
        messages=[m.model_dump() for m in body.messages] if isinstance(body.messages, list) else body.messages,
        params=body.params.model_dump(),
        stream=False,
    )

    final_res = {
            "text": result["text"],
            "metrics": result.get("metrics", {
                "ttft_ms": None,
                "latency_ms": 0.0,
                "model": body.model_id,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": None
            }),
        }

    return ChatResponse.model_validate(final_res)

@mlflow.trace()
@app.post(
    "/chat.stream",
    response_model=None,  # streaming is not a Pydantic field
    responses={
        200: {
            "content": {"text/plain": {}},
            "description": "Streaming of raw text chunks (inline; clients append to the same line)",
        }
    },
)
def chat_stream(body: ChatRequest):
    gen = route_call(
        provider_id=body.provider_id,
        model_id=body.model_id,
        messages=[m.model_dump() for m in body.messages] if isinstance(body.messages, list) else body.messages,
        params=body.params.model_dump(),
        stream=True,
        stream_mode="raw"
    )

    return StreamingResponse(gen, media_type="text/plain")

# ==========================================================================
# ==========================================================================
@mlflow.trace()
@app.post("/chat.streamsse", response_class=StreamingResponse)
def chat_stream_sse(body: ChatRequest):
    gen = route_call(
        provider_id=body.provider_id,
        model_id=body.model_id,
        messages=[m.model_dump() for m in body.messages] if isinstance(body.messages, list) else body.messages,
        params=body.params.model_dump(),
        stream=True,
        stream_mode="sse",              # emits already-framed SSE bytes
    )
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
# ==========================================================================
# ==========================================================================

@app.exception_handler(BackendError)
def handle_backend_error(request: Request, exc: BackendError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message},
    )


@app.exception_handler(ValueError)
def handle_value_error(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)},
    )
