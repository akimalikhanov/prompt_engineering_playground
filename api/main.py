from fastapi import FastAPI, HTTPException, Query, Response, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Generator, Optional
from utils.errors import BackendError
from utils.sse import sse_pack
from utils.load_configs import _load_models_config
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
    get_latest_by_key,
    get_prompt_by_id_and_version,
    update_prompt_status
)
from utils.logging import setup_logging, correlation_id_middleware, get_correlation_id
from utils.jinja_renderer import render_messages
from dotenv import load_dotenv
import os
import mlflow
import json
import logging

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
            context_prompt=body.context_prompt,
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
            context_prompt=body.context_prompt,
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
                    context_prompt=body.context_prompt,
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
                    context_prompt=body.context_prompt,
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
            context_prompt=body.context_prompt,
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
                            line = line.strip()
                            if line.startswith('data: '):
                                # Extract data content (skip 'data: ' prefix)
                                data_content = line[6:]
                                if data_content:  # Only add non-empty content
                                    accumulated_text += data_content
                    
                    # Extract metrics from metrics event
                    if 'event: metrics' in chunk_str:
                        lines = chunk_str.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line.startswith('data: '):
                                try:
                                    json_str = line[6:]  # Skip 'data: ' prefix
                                    if json_str:
                                        metrics = json.loads(json_str)
                                except (json.JSONDecodeError, ValueError):
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
                    context_prompt=body.context_prompt,
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
                    context_prompt=body.context_prompt,
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
            context_prompt=body.context_prompt,
        )
        raise


# ============================================
# Prompts Management Endpoints
# ============================================

@app.get("/prompts", response_model=ListPromptsResponse)
def list_prompts(
    technique: Optional[str] = Query(None, description="Filter by technique ('zero_shot', 'few_shot', 'prompt_chain')"),
    q: Optional[str] = Query(None, description="Search query for title/description"),
    is_active: Optional[bool] = Query(None, description="Filter by active status")
):
    """
    List latest prompts from v_prompt_examples_latest view.
    
    Query params:
    - technique: Filter by technique ('zero_shot', 'few_shot', 'prompt_chain')
    - q: Search in title and description
    - is_active: Filter by is_active flag
    """
    try:
        prompts = get_latest_prompts(
            technique=technique,
            q=q,
            is_active=is_active
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
    Create a new prompt with version = 1, is_active = true.
    
    Body:
    - key: Internal key (unique identifier)
    - title: Prompt title
    - prompt_template: Messages array in LLM format with Jinja variables
    - technique: Technique type ('zero_shot', 'few_shot', 'prompt_chain')
    - description: Optional description
    - category: Optional category
    - tags: Optional list of tags
    - variables: Optional list of variable definitions
    - default_examples: Optional few-shot examples
    - response_format: Optional response format ('json_object', 'json_schema')
    - json_schema_template: Optional JSON schema template (only if response_format='json_schema')
    """
    try:
        # Convert Pydantic models to dicts
        prompt_template = [msg.model_dump() for msg in body.prompt_template] if hasattr(body, 'prompt_template') and body.prompt_template else []
        variables_dicts = [var.model_dump() for var in body.variables] if body.variables else []
        
        prompt = create_prompt(
            key=body.key,
            title=body.title,
            prompt_template=prompt_template,
            technique=body.technique,
            description=getattr(body, 'description', None),
            category=getattr(body, 'category', None),
            tags=getattr(body, 'tags', None),
            variables=variables_dicts,
            default_examples=getattr(body, 'default_examples', None),
            response_format=getattr(body, 'response_format', None),
            json_schema_template=getattr(body, 'json_schema_template', None)
        )
        
        return PromptExampleResponse.model_validate(prompt)
    
    except Exception as e:
        logger.error(f"Error creating prompt: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/prompts/{prompt_id}", response_model=PromptExampleResponse)
def get_prompt(prompt_id: str):
    """
    Get a specific prompt by its id (any version).
    
    Path params:
    - prompt_id: UUID of the prompt
    """
    try:
        prompt = get_prompt_by_id(prompt_id)
        
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        return PromptExampleResponse.model_validate(prompt)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt {prompt_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prompts/{prompt_id}/versions", response_model=PromptExampleResponse, status_code=201)
def create_version(prompt_id: str, body: CreateVersionRequest):
    """
    Create a new version of an existing prompt.
    
    Clones the key, increments version, sets is_active=true.
    Optionally overrides prompt_template, variables, and other fields.
    
    Path params:
    - prompt_id: UUID of the base prompt to clone
    
    Body (all optional):
    - prompt_template: Override prompt template (messages array)
    - variables: Override variables
    - default_examples: Override default examples
    - response_format: Override response format
    - json_schema_template: Override JSON schema template
    - description: Override description
    - category: Override category
    - tags: Override tags
    """
    try:
        # Convert Pydantic models to dicts if provided
        prompt_template = None
        if hasattr(body, 'prompt_template') and body.prompt_template:
            prompt_template = [msg.model_dump() for msg in body.prompt_template]
        
        variables_dicts = None
        if hasattr(body, 'variables') and body.variables:
            variables_dicts = [var.model_dump() for var in body.variables]
        
        # Create new version (auto-deactivate previous)
        new_prompt = create_new_version(
            prompt_id=prompt_id,
            prompt_template=prompt_template,
            variables=variables_dicts,
            default_examples=getattr(body, 'default_examples', None),
            response_format=getattr(body, 'response_format', None),
            json_schema_template=getattr(body, 'json_schema_template', None),
            description=getattr(body, 'description', None),
            category=getattr(body, 'category', None),
            tags=getattr(body, 'tags', None),
            auto_deactivate_previous=True  # Deactivate previous active version
        )
        
        if not new_prompt:
            raise HTTPException(status_code=404, detail="Base prompt not found")
        
        return PromptExampleResponse.model_validate(new_prompt)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating version for {prompt_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/prompts/{key}/latest", response_model=PromptExampleResponse)
def get_latest_prompt(key: str):
    """
    Get the latest active prompt by key.
    
    Path params:
    - key: Prompt key (unique identifier)
    """
    try:
        prompt = get_latest_by_key(key)
        
        if not prompt:
            raise HTTPException(
                status_code=404,
                detail=f"No active prompt found for key '{key}'"
            )
        
        return PromptExampleResponse.model_validate(prompt)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest prompt {key}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prompts/{prompt_id}/render", response_model=RenderPromptResponse)
def render_prompt(
    prompt_id: str,
    body: RenderPromptRequest,
    version: Optional[int] = Query(None, description="Specific version to render (default: latest)")
):
    """
    Render prompt messages with Jinja2 using provided variables.
    
    Path params:
    - prompt_id: UUID of the prompt
    
    Query params:
    - version: Optional version number (default: uses the prompt_id's version)
    
    Body:
    - variables: Dict of variables to substitute in templates
    
    Returns:
    - rendered_messages: List of messages with rendered content
    - used_version: Version number that was rendered
    - missing_vars: List of required variables that were not provided
    - warnings: List of warnings (e.g., unused variables, template errors)
    """
    try:
        # Get the prompt (specific version if provided, otherwise the one with this prompt_id)
        prompt = get_prompt_by_id_and_version(prompt_id, version)
        
        if not prompt:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt not found for prompt_id={prompt_id}" + 
                       (f", version={version}" if version else "")
            )
        
        # Render messages
        rendered_msgs, missing_vars, warnings = render_messages(
            messages=prompt.prompt_template,
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
        logger.error(f"Error rendering prompt {prompt_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/prompts/{prompt_id}", response_model=PromptExampleResponse)
def update_prompt(prompt_id: str, body: PatchPromptRequest):
    """
    Update prompt's is_active flag.
    
    Path params:
    - prompt_id: UUID of the prompt
    
    Body:
    - is_active: Set active flag (true/false)
    """
    try:
        # Check that is_active is provided
        if body.is_active is None:
            raise HTTPException(
                status_code=400,
                detail="'is_active' must be provided"
            )
        
        updated_prompt = update_prompt_status(
            prompt_id=prompt_id,
            is_active=body.is_active
        )
        
        if not updated_prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        return PromptExampleResponse.model_validate(updated_prompt)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prompt {prompt_id}: {e}", exc_info=True)
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
