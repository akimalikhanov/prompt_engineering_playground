from fastapi import FastAPI, HTTPException, Query, Response, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Generator, Optional
from utils.errors import BackendError
from utils.sse import sse_pack
from config.load_models_config import _load_models_config
from schemas.schemas import *
from schemas.prompts import (
    CreatePromptRequest,
    CreateVersionRequest,
    RenderPromptRequest,
    PatchPromptRequest,
    PromptExampleResponse,
    ListPromptsResponse,
    RenderPromptResponse,
    PromptMessage
)
from services.router import route_call
from services.runs_logger import log_run, get_run_by_id
from services.prompts_service import (
    get_latest_prompts,
    create_prompt,
    get_prompt_by_id,
    create_new_version,
    get_latest_by_technique_and_title,
    get_prompt_by_id_and_version,
    update_prompt_status
)
from utils.logging import setup_logging, correlation_id_middleware, get_correlation_id
from utils.jinja_renderer import render_messages
from dotenv import load_dotenv
import os
import mlflow
import json

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

# Ensure tracing is enabled
try:
    mlflow.tracing.enable()
except Exception:
    pass  # Tracing might already be enabled or not available


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

@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={200: {"description": "Complete response as JSON"}},
)
# @mlflow.trace()
def chat(body: ChatRequest):
    """Non-streaming chat endpoint with runs logging."""
    trace_id = get_correlation_id()

    try:
        # Do NOT start any spans if you want exactly one span total.
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

        log_run(
            trace_id=trace_id,
            provider_key=body.provider_id,
            model_id=body.model_id,
            messages=body.messages,
            params=body.params.model_dump(),
            output_text=result["text"],
            metrics=result.get("metrics", {}),
            status="ok",
        )
        return ChatResponse.model_validate(final_res)

    except Exception as e:
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
    
    # For streaming endpoints, we don't use @mlflow.trace() decorator
    # because it causes issues when the function returns a generator.
    # MLflow autolog will still create traces for the underlying API calls.
    
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
                log_run(
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

@app.post("/chat.streamsse", response_class=StreamingResponse)
def chat_stream_sse(body: ChatRequest):
    """SSE streaming chat endpoint with runs logging."""
    trace_id = get_correlation_id()
    
    # For streaming endpoints, we don't use @mlflow.trace() decorator
    # because it causes issues when the function returns a generator.
    # MLflow autolog will still create traces for the underlying API calls.
    
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
                log_run(
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


# ============================================
# Prompts Management Endpoints
# ============================================

@app.get("/prompts", response_model=ListPromptsResponse)
def list_prompts(
    technique_key: Optional[str] = Query(None, description="Filter by technique key"),
    q: Optional[str] = Query(None, description="Search query for title/technique"),
    enabled: Optional[bool] = Query(None, description="Filter by enabled status")
):
    """
    List latest prompts from v_prompt_examples_latest view.
    
    Query params:
    - technique_key: Filter by technique (e.g., 'cot', 'react')
    - q: Search in title and technique_key
    - enabled: Filter by is_enabled flag
    """
    try:
        prompts = get_latest_prompts(
            technique_key=technique_key,
            q=q,
            enabled=enabled
        )
        
        return ListPromptsResponse(
            prompts=[PromptExampleResponse.model_validate(p) for p in prompts],
            count=len(prompts)
        )
    except Exception as e:
        logger.error(f"Error listing prompts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prompts", response_model=PromptExampleResponse, status_code=201)
def create_prompt_endpoint(body: CreatePromptRequest):
    """
    Create a new prompt with version = 1, status = 'active', is_enabled = true.
    
    Body:
    - technique_key: Technique identifier (must exist)
    - title: Prompt title
    - language: Language code (default 'en')
    - messages: List of message objects with role and content
    - variables: Optional list of variable definitions
    - model_hint: Optional model recommendation
    """
    try:
        # Convert Pydantic models to dicts
        messages_dicts = [msg.model_dump() for msg in body.messages]
        variables_dicts = [var.model_dump() for var in body.variables] if body.variables else []
        
        prompt = create_prompt(
            technique_key=body.technique_key,
            title=body.title,
            messages=messages_dicts,
            language=body.language or "en",
            variables=variables_dicts,
            model_hint=body.model_hint
        )
        
        return PromptExampleResponse.model_validate(prompt)
    
    except Exception as e:
        logger.error(f"Error creating prompt: {e}", exc_info=True)
        # Check if it's a foreign key constraint error (technique doesn't exist)
        if "foreign key constraint" in str(e).lower():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid technique_key: {body.technique_key} does not exist"
            )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/prompts/{example_id}", response_model=PromptExampleResponse)
def get_prompt(example_id: str):
    """
    Get a specific prompt by its example_id (any version).
    
    Path params:
    - example_id: UUID of the prompt
    """
    try:
        prompt = get_prompt_by_id(example_id)
        
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        return PromptExampleResponse.model_validate(prompt)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt {example_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prompts/{example_id}/versions", response_model=PromptExampleResponse, status_code=201)
def create_version(example_id: str, body: CreateVersionRequest):
    """
    Create a new version of an existing prompt.
    
    Clones the (technique_key, title), increments version, sets status='active'.
    Optionally overrides messages, variables, model_hint, or language.
    
    Path params:
    - example_id: UUID of the base prompt to clone
    
    Body (all optional):
    - messages: Override messages
    - variables: Override variables
    - model_hint: Override model hint
    - language: Override language
    """
    try:
        # Convert Pydantic models to dicts if provided
        messages_dicts = None
        if body.messages:
            messages_dicts = [msg.model_dump() for msg in body.messages]
        
        variables_dicts = None
        if body.variables:
            variables_dicts = [var.model_dump() for var in body.variables]
        
        # Create new version (auto-archive previous)
        new_prompt = create_new_version(
            example_id=example_id,
            messages=messages_dicts,
            variables=variables_dicts,
            model_hint=body.model_hint,
            language=body.language,
            auto_archive_previous=True  # Archive previous active version
        )
        
        if not new_prompt:
            raise HTTPException(status_code=404, detail="Base prompt not found")
        
        return PromptExampleResponse.model_validate(new_prompt)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating version for {example_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/prompts/{technique_key}/{title}/latest", response_model=PromptExampleResponse)
def get_latest_prompt(technique_key: str, title: str):
    """
    Get the latest active and enabled prompt by technique_key and title.
    
    Path params:
    - technique_key: Technique identifier
    - title: Prompt title
    """
    try:
        prompt = get_latest_by_technique_and_title(technique_key, title)
        
        if not prompt:
            raise HTTPException(
                status_code=404,
                detail=f"No active prompt found for technique '{technique_key}' and title '{title}'"
            )
        
        return PromptExampleResponse.model_validate(prompt)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest prompt {technique_key}/{title}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prompts/{example_id}/render", response_model=RenderPromptResponse)
def render_prompt(
    example_id: str,
    body: RenderPromptRequest,
    version: Optional[int] = Query(None, description="Specific version to render (default: latest)")
):
    """
    Render prompt messages with Jinja2 using provided variables.
    
    Path params:
    - example_id: UUID of the prompt
    
    Query params:
    - version: Optional version number (default: uses the example_id's version)
    
    Body:
    - variables: Dict of variables to substitute in templates
    
    Returns:
    - rendered_messages: List of messages with rendered content
    - used_version: Version number that was rendered
    - missing_vars: List of required variables that were not provided
    - warnings: List of warnings (e.g., unused variables, template errors)
    """
    try:
        # Get the prompt (specific version if provided, otherwise the one with this example_id)
        prompt = get_prompt_by_id_and_version(example_id, version)
        
        if not prompt:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt not found for example_id={example_id}" + 
                       (f", version={version}" if version else "")
            )
        
        # Render messages
        rendered_msgs, missing_vars, warnings = render_messages(
            messages=prompt.messages,
            variables=body.variables
        )
        
        # Convert to PromptMessage objects
        rendered_messages = [
            PromptMessage(role=msg['role'], content=msg['content'])
            for msg in rendered_msgs
        ]
        
        return RenderPromptResponse(
            rendered_messages=rendered_messages,
            used_version=prompt.version,
            missing_vars=missing_vars,
            warnings=warnings
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rendering prompt {example_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/prompts/{example_id}", response_model=PromptExampleResponse)
def update_prompt(example_id: str, body: PatchPromptRequest):
    """
    Update prompt's is_enabled flag and/or status.
    
    Path params:
    - example_id: UUID of the prompt
    
    Body (at least one required):
    - is_enabled: Set enabled flag (true/false)
    - status: Set status ('active', 'archived', 'draft')
    """
    try:
        # Check that at least one field is provided
        if body.is_enabled is None and body.status is None:
            raise HTTPException(
                status_code=400,
                detail="At least one of 'is_enabled' or 'status' must be provided"
            )
        
        updated_prompt = update_prompt_status(
            example_id=example_id,
            is_enabled=body.is_enabled,
            status=body.status
        )
        
        if not updated_prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        return PromptExampleResponse.model_validate(updated_prompt)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prompt {example_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Runs Endpoints
# ============================================

@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: int):
    """
    Get a specific run by its ID from app.runs table.
    
    Path params:
    - run_id: The ID of the run (BigInteger primary key)
    """
    try:
        run = get_run_by_id(run_id)
        
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        
        return RunResponse.model_validate(run)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Exception Handlers
# ============================================

@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    """
    Log validation errors to database before returning 422 response.
    This captures malformed requests for analytics and debugging.
    """
    trace_id = get_correlation_id()
    
    # Try to extract request body for logging
    provider_id = "unknown"
    model_id = "unknown"
    messages = None
    params = {}
    
    try:
        # Get the raw request body
        body = await request.body()
        if body:
            body_json = json.loads(body.decode('utf-8'))
            provider_id = body_json.get("provider_id", "unknown")
            model_id = body_json.get("model_id", "unknown")
            messages = body_json.get("messages")
            params = body_json.get("params", {})
    except Exception:
        # If we can't parse the body, just continue with defaults
        pass
    
    # Format validation errors for logging
    error_details = []
    for error in exc.errors():
        loc = " -> ".join(str(x) for x in error["loc"])
        error_details.append(f"{loc}: {error['msg']}")
    error_message = "; ".join(error_details)
    
    # Log validation error to database
    log_run(
        trace_id=trace_id,
        provider_key=provider_id,
        model_id=model_id,
        messages=messages,
        params=params,
        status="error",
        error_type="RequestValidationError",
        error_message=error_message,
    )
    
    # Return standard FastAPI validation error response
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


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
