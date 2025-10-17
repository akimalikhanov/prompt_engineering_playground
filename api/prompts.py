"""
FastAPI endpoints for prompt management
Copy of api/main.py structure with prompt-specific endpoints
"""
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Optional
from dotenv import load_dotenv
import os

from utils.logging import setup_logging, correlation_id_middleware
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
from services.prompts_service import (
    get_latest_prompts,
    create_prompt,
    get_prompt_by_id,
    create_new_version,
    get_latest_by_technique_and_title,
    get_prompt_by_id_and_version,
    update_prompt_status
)
from utils.jinja_renderer import render_messages

load_dotenv()

app = FastAPI(title="Prompt Engineering Playground - Prompts API")

logger = setup_logging(
    app_logger_name=os.getenv("LOGGER_NAME", "llm-router"),
    level=20,
    to_console=True,
    to_file=True,
    file_path="logs/app.log",
    hijack_uvicorn=True,
)

app.middleware("http")(correlation_id_middleware)


# ============================================
# Health check endpoints (from main.py)
# ============================================
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Prompts API"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================
# Endpoint 1: List prompts (latest only)
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


# ============================================
# Endpoint 2: Create a prompt (v1)
# ============================================
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


# ============================================
# Endpoint 3: Get a prompt
# ============================================
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


# ============================================
# Endpoint 4: Create a new version
# ============================================
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


# ============================================
# Endpoint 5: Fetch latest by key+title
# ============================================
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


# ============================================
# Endpoint 6: Render (Jinja over each message.content)
# ============================================
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


# ============================================
# Endpoint 7: Enable/disable or archive
# ============================================
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
# Exception handlers (from main.py)
# ============================================
@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    error_details = []
    for error in exc.errors():
        loc = " -> ".join(str(x) for x in error["loc"])
        error_details.append(f"{loc}: {error['msg']}")
    error_message = "; ".join(error_details)
    
    logger.error(f"Validation error: {error_message}")
    
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.exception_handler(ValueError)
def handle_value_error(request: Request, exc: ValueError):
    """Handle value errors."""
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)},
    )

