"""
Example of API with runs logging integrated.
This is a reference implementation - you can merge this into your main.py
"""
from fastapi import FastAPI, HTTPException, Query, Response, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Generator
from utils.errors import BackendError
from utils.sse import sse_pack
from utils.logging import setup_logging, correlation_id_middleware, get_correlation_id
from config.load_models_config import _load_models_config
from schemas.schemas import *
from services.router import route_call
from services.runs_logger import log_run, log_streaming_run
from dotenv import load_dotenv
import os
import mlflow
import json

load_dotenv()

# MLflow and OTEL setup (same as before)
os.environ.setdefault("MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT", "true")
os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces")
os.environ.setdefault("OTEL_RESOURCE_ATTRIBUTES", "service.name=pep-api,service.namespace=pep,deployment.environment=prod")
os.environ.setdefault("OTEL_SERVICE_NAME", "pep-api")
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
    level=20,
    to_console=True,
    to_file=True,
    file_path="logs/app.log",
    hijack_uvicorn=True,
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
    """Non-streaming chat endpoint with runs logging."""
    trace_id = get_correlation_id()
    
    try:
        result = route_call(
            provider_id=body.provider_id,
            model_id=body.model_id,
            messages=[m.model_dump() for m in body.messages] if isinstance(body.messages, list) else body.messages,
            params=body.params.model_dump(),
            stream=False,
        )

        final_res = {
            "text": result["text"],
            "usage": result.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, 'total_tokens': 0}),
            "metrics": result.get("metrics", {"ttft_ms": None, "latency_ms": 0.0, "model": body.model_id}),
        }

        # Log successful run to database
        log_run(
            trace_id=trace_id,
            provider_key=body.provider_id,
            model_id=body.model_id,
            messages=body.messages,
            params=body.params.model_dump(),
            output_text=result["text"],
            prompt_tokens=result.get("usage", {}).get("prompt_tokens"),
            completion_tokens=result.get("usage", {}).get("completion_tokens"),
            total_tokens=result.get("usage", {}).get("total_tokens"),
            cost_usd=result.get("metrics", {}).get("cost_usd"),
            latency_ms=int(result.get("metrics", {}).get("latency_ms", 0)),
            ttft_ms=None,  # Not available for non-streaming
            status="ok",
        )

        return ChatResponse.model_validate(final_res)
    
    except Exception as e:
        # Log failed run to database
        log_run(
            trace_id=trace_id,
            provider_key=body.provider_id,
            model_id=body.model_id,
            messages=body.messages,
            params=body.params.model_dump(),
            status="error",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        raise


@mlflow.trace()
@app.post(
    "/chat.stream",
    response_model=None,
    responses={
        200: {
            "content": {"text/plain": {}},
            "description": "Streaming of raw text chunks",
        }
    },
)
def chat_stream(body: ChatRequest):
    """Streaming chat endpoint with runs logging."""
    trace_id = get_correlation_id()
    
    try:
        gen = route_call(
            provider_id=body.provider_id,
            model_id=body.model_id,
            messages=[m.model_dump() for m in body.messages] if isinstance(body.messages, list) else body.messages,
            params=body.params.model_dump(),
            stream=True,
            stream_mode="raw"
        )

        # Wrap generator to collect output and metrics
        def logged_generator():
            accumulated_text = ""
            metrics = {}
            
            try:
                for chunk in gen:
                    # Check if this is the metrics chunk (JSON at the end)
                    if chunk.startswith("\n{") and "metrics" in chunk:
                        try:
                            metrics_obj = json.loads(chunk.strip())
                            metrics = metrics_obj.get("metrics", {})
                        except json.JSONDecodeError:
                            pass
                    else:
                        accumulated_text += chunk
                    
                    yield chunk
                
                # Log after streaming completes
                log_streaming_run(
                    trace_id=trace_id,
                    provider_key=body.provider_id,
                    model_id=body.model_id,
                    messages=body.messages,
                    params=body.params.model_dump(),
                    accumulated_text=accumulated_text,
                    metrics=metrics,
                    status="ok",
                )
            except Exception as e:
                # Log error
                log_run(
                    trace_id=trace_id,
                    provider_key=body.provider_id,
                    model_id=body.model_id,
                    messages=body.messages,
                    params=body.params.model_dump(),
                    output_text=accumulated_text if accumulated_text else None,
                    status="error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                raise

        return StreamingResponse(logged_generator(), media_type="text/plain")
    
    except Exception as e:
        # Log error if it happens before streaming starts
        log_run(
            trace_id=trace_id,
            provider_key=body.provider_id,
            model_id=body.model_id,
            messages=body.messages,
            params=body.params.model_dump(),
            status="error",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        raise


@mlflow.trace()
@app.post("/chat.streamsse", response_class=StreamingResponse)
def chat_stream_sse(body: ChatRequest):
    """SSE streaming chat endpoint with runs logging."""
    trace_id = get_correlation_id()
    
    try:
        gen = route_call(
            provider_id=body.provider_id,
            model_id=body.model_id,
            messages=[m.model_dump() for m in body.messages] if isinstance(body.messages, list) else body.messages,
            params=body.params.model_dump(),
            stream=True,
            stream_mode="sse",
        )

        def logged_generator():
            accumulated_text = ""
            metrics = {}
            
            try:
                for chunk in gen:
                    # Parse SSE events to extract data
                    chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                    
                    # Extract text from message events
                    if 'event: message' in chunk_str:
                        lines = chunk_str.split('\n')
                        for line in lines:
                            if line.startswith('data: '):
                                accumulated_text += line[6:]
                    
                    # Extract metrics from metrics event
                    elif 'event: metrics' in chunk_str:
                        lines = chunk_str.split('\n')
                        for line in lines:
                            if line.startswith('data: '):
                                try:
                                    metrics = json.loads(line[6:])
                                except json.JSONDecodeError:
                                    pass
                    
                    yield chunk
                
                # Log after streaming completes
                log_streaming_run(
                    trace_id=trace_id,
                    provider_key=body.provider_id,
                    model_id=body.model_id,
                    messages=body.messages,
                    params=body.params.model_dump(),
                    accumulated_text=accumulated_text,
                    metrics=metrics,
                    status="ok",
                )
            except Exception as e:
                # Log error
                log_run(
                    trace_id=trace_id,
                    provider_key=body.provider_id,
                    model_id=body.model_id,
                    messages=body.messages,
                    params=body.params.model_dump(),
                    output_text=accumulated_text if accumulated_text else None,
                    status="error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                raise

        return StreamingResponse(
            logged_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    
    except Exception as e:
        # Log error if it happens before streaming starts
        log_run(
            trace_id=trace_id,
            provider_key=body.provider_id,
            model_id=body.model_id,
            messages=body.messages,
            params=body.params.model_dump(),
            status="error",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        raise


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


