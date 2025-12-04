"""
Service for logging LLM runs to app.runs table
"""
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
from utils.errors import BackendError, _as_backend_error
from utils.otel_metrics import record_runs_feedback
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.runs import Run

# Load environment variables
# env_path = Path(__file__).parent.parent / ".env"
load_dotenv()

# Database connection
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5433")
DB_NAME = os.getenv("APP_DB", "app")
DB_USER = os.getenv("APP_DB_USER", "app_user")
DB_PASSWORD = os.getenv("APP_DB_PASSWORD", "app_password")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine and session factory
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def extract_input_text(messages: Any, max_length: int = 2000) -> Optional[str]:
    """Extract sanitized input text from messages."""
    try:
        if isinstance(messages, str):
            return messages[:max_length]
        elif isinstance(messages, list):
            last_user_content: Optional[str] = None
            last_any_content: Optional[str] = None
            for msg in messages:
                content = None
                role = "user"
                if isinstance(msg, dict):
                    role = msg.get("role", role)
                    content = msg.get("content")
                else:
                    # Support Pydantic models (e.g., ChatMessage) or objects with attributes
                    role = getattr(msg, "role", role)
                    content = getattr(msg, "content", None)
                if content:
                    last_any_content = content
                    if role == "user":
                        last_user_content = content
            chosen = last_user_content or last_any_content
            return chosen[:max_length] if chosen else None
    except Exception:
        pass
    return None


def _extract_first_message(
    messages: Any,
    *,
    role: Optional[str],
    prefix: Optional[str] = None,
    strip_prefix: bool = False,
    max_length: int = 4000,
) -> Optional[str]:
    try:
        if isinstance(messages, list):
            for msg in messages:
                msg_role: Optional[str]
                content: Any
                if isinstance(msg, dict):
                    msg_role = msg.get("role")
                    content = msg.get("content")
                else:
                    msg_role = getattr(msg, "role", None)
                    content = getattr(msg, "content", None)
                if role and msg_role != role:
                    continue
                if not content:
                    continue
                if prefix:
                    if not isinstance(content, str) or not content.startswith(prefix):
                        continue
                    content_value = content[len(prefix) :] if strip_prefix else content
                else:
                    content_value = content
                if isinstance(content_value, str):
                    return content_value[:max_length]
                return None
    except Exception:
        pass
    return None


def extract_system_prompt(messages: Any, max_length: int = 4000) -> Optional[str]:
    """Extract the first system prompt from the messages payload."""
    return _extract_first_message(messages, role="system", max_length=max_length)


def extract_context_prompt(messages: Any, max_length: int = 4000) -> Optional[str]:
    """Extract the first CONTEXT user message."""
    return _extract_first_message(
        messages,
        role="user",
        prefix="CONTEXT:\n",
        strip_prefix=True,
        max_length=max_length,
    )


def _log_chat_failure(
    exc: Exception,
    *,
    trace_id: str,
    body: Any,
    session_id: Optional[str],
    accumulated_text: Optional[str] = None,
) -> BackendError:
    """
    Normalize an exception into BackendError and log a failed chat run.
    Shared by streaming/non-streaming chat endpoints.
    """
    normalized = _as_backend_error(exc)
    log_run(
        trace_id=trace_id,
        provider_key=getattr(body, "provider_id", None),
        model_id=getattr(body, "model_id", None),
        messages=getattr(body, "messages", None),
        params=getattr(body, "params", {}).model_dump()
        if hasattr(getattr(body, "params", None), "model_dump")
        else getattr(body, "params", {}),
        output_text=accumulated_text if accumulated_text else None,
        status="error",
        error_type=type(exc).__name__,
        error_message=normalized.message,
        context_prompt=getattr(body, "context_prompt", None),
        session_id=session_id,
    )
    return normalized


def log_run(
    trace_id: str,
    provider_key: str,
    model_id: str,
    messages: Any,
    params: Dict[str, Any],
    system_prompt: Optional[str] = None,
    context_prompt: Optional[str] = None,
    output_text: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
    latency_ms: Optional[int] = None,
    ttft_ms: Optional[int] = None,
    tokens_per_second: Optional[float] = None,
    status: str = "ok",
    error_type: Optional[str] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    prompt_key: Optional[str] = None,
    prompt_version: Optional[str] = None,
    technique_key: Optional[str] = None,
    variables_json: Optional[List[Dict]] = None,
    metadata: Optional[Dict] = None,
    pricing_snapshot: Optional[Dict] = None,
    tool_call: Optional[Any] = None,
    # Dict-based parameters (for convenience)
    metrics: Optional[Dict[str, Any]] = None,
    error_info: Optional[Dict[str, str]] = None,
    accumulated_text: Optional[str] = None,  # Alias for output_text (for streaming)
) -> Optional[int]:
    """
    Log an LLM run to the database.
    
    Supports two calling styles:
    1. Individual parameters (backward compatible)
    2. Dict-based parameters via 'metrics' and 'error_info' (convenient for streaming)
    
    Args:
        trace_id: Correlation/trace ID for the request
        provider_key: Provider identifier (e.g., 'openai', 'google')
        model_id: Model identifier (e.g., 'gpt', 'gemini')
        messages: Input messages (string or list)
        params: Model parameters (temperature, top_p, etc.)
        system_prompt: System prompt/instructions applied to the run
        context_prompt: Retrieved or supplemental context sent with the run
        output_text: Model output text
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total tokens used
        cost_usd: Cost in USD
        latency_ms: Total latency in milliseconds
        ttft_ms: Time to first token in milliseconds
        tokens_per_second: Tokens generated per second
        status: Run status ('ok', 'error', 'rate_limited', 'timeout', 'cancelled')
        error_type: Error type if status is error
        error_code: Error code if status is error
        error_message: Error message if status is error
        request_id: Unique request ID
        session_id: Session/conversation ID
        user_id: User identifier
        prompt_key: Prompt template key
        prompt_version: Prompt version/tag
        technique_key: Technique used (e.g., 'cot', 'react')
        variables_json: Template variables used
        metadata: Additional metadata as JSON
        pricing_snapshot: Pricing information snapshot
        tool_call: Executed tool information to persist to the `tool_call` JSONB column.
        metrics: Dict containing metrics (prompt_tokens, completion_tokens, total_tokens, 
                 cost_usd, latency_ms, ttft_ms, tokens_per_second). Values from this dict are used if 
                 individual parameters are not provided.
        error_info: Dict containing error details (type, code, message). Values from this
                    dict are used if individual error parameters are not provided.
        accumulated_text: Alias for output_text (used in streaming contexts)
    
    Returns:
        The ID of the created run record, or None if failed
    """
    # Extract values from metrics dict if provided and individual params are None
    if metrics:
        if prompt_tokens is None:
            prompt_tokens = metrics.get("prompt_tokens")
        if completion_tokens is None:
            completion_tokens = metrics.get("completion_tokens")
        if total_tokens is None:
            total_tokens = metrics.get("total_tokens")
        if cost_usd is None:
            cost_usd = metrics.get("cost_usd")
        if latency_ms is None:
            latency_ms = int(metrics.get("latency_ms", 0)) if metrics.get("latency_ms") else None
        if ttft_ms is None:
            ttft_ms = int(metrics.get("ttft_ms", 0)) if metrics.get("ttft_ms") else None
        if tokens_per_second is None:
            tokens_per_second = metrics.get("tokens_per_second")
        # If tool_call was not passed explicitly, try to pull it from metrics
        if tool_call is None:
            tool_call = metrics.get("executed_tools")
    
    # Extract values from error_info dict if provided and individual params are None
    if error_info:
        if error_type is None:
            error_type = error_info.get("type")
        if error_code is None:
            error_code = error_info.get("code")
        if error_message is None:
            error_message = error_info.get("message")
    
    # Use accumulated_text as alias for output_text if output_text is not provided
    if output_text is None and accumulated_text is not None:
        output_text = accumulated_text
    
    session = SessionLocal()
    try:
        run = Run(
            trace_id=trace_id,
            request_id=request_id or trace_id,  # Use trace_id as fallback
            session_id=session_id,
            user_id=user_id,
            provider_key=provider_key,
            model_id=model_id,
            prompt_key=prompt_key,
            prompt_version=prompt_version,
            technique_key=technique_key,
            params_json=params or {},
            variables_json=variables_json or [],
            input_text=extract_input_text(messages),
            system_prompt=(system_prompt or extract_system_prompt(messages)),
            context_prompt=(context_prompt or extract_context_prompt(messages)),
            output_text=output_text[:2000] if output_text else None,  # Truncate for preview
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tokens_per_second=tokens_per_second,
            status=status,
            error_type=error_type,
            error_code=error_code,
            error_message=error_message[:500] if error_message else None,
            pricing_snapshot=pricing_snapshot or {},
            metadata_json=metadata or {},
            tool_call=tool_call,
        )
        
        session.add(run)
        session.commit()
        session.refresh(run)
        return run.id
    except Exception as e:
        session.rollback()
        # Log error but don't fail the request
        import logging
        logger = logging.getLogger(os.getenv("LOGGER_NAME", "llm-router"))
        logger.error(f"Failed to log run to database: {e}", exc_info=True)
        return None
    finally:
        session.close()


def get_run_by_id(run_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a run by its ID from the database.
    
    Args:
        run_id: The ID of the run to retrieve
    
    Returns:
        A dictionary containing the run data, or None if not found
    """
    session = SessionLocal()
    try:
        run = session.query(Run).filter(Run.id == run_id).first()
        
        if not run:
            return None
        
        # Convert SQLAlchemy model to dict
        run_dict = {
            "id": run.id,
            "occurred_at": run.occurred_at.isoformat() if run.occurred_at else None,
            "trace_id": run.trace_id,
            "request_id": run.request_id,
            "session_id": run.session_id,
            "user_id": run.user_id,
            "provider_key": run.provider_key,
            "model_id": run.model_id,
            "prompt_key": run.prompt_key,
            "prompt_version": run.prompt_version,
            "technique_key": run.technique_key,
            "params_json": run.params_json or {},
            "variables_json": run.variables_json or [],
            "input_text": run.input_text,
            "system_prompt": run.system_prompt,
            "context_prompt": run.context_prompt,
            "output_text": run.output_text,
            "output_preview": run.output_preview,
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "total_tokens": run.total_tokens,
            "reasoning_tokens": run.reasoning_tokens,
            "cost_usd": float(run.cost_usd) if run.cost_usd is not None else None,
            "latency_ms": run.latency_ms,
            "ttft_ms": run.ttft_ms,
            "tokens_per_second": float(run.tokens_per_second) if run.tokens_per_second is not None else None,
            "status": run.status,
            "error_type": run.error_type,
            "error_code": run.error_code,
            "error_message": run.error_message,
            "cached": run.cached,
            "pricing_snapshot": run.pricing_snapshot or {},
            "metadata_json": run.metadata_json or {},
            "tool_call": run.tool_call,
        }
        
        return run_dict
    except Exception as e:
        import logging
        logger = logging.getLogger(os.getenv("LOGGER_NAME", "llm-router"))
        logger.error(f"Failed to get run {run_id} from database: {e}", exc_info=True)
        return None
    finally:
        session.close()


def update_run_feedback_by_trace_id(trace_id: str, user_feedback: int) -> bool:
    """
    Update user_feedback for the most recent run with the given trace_id.
    Returns True if a run was updated, False if no matching run was found.
    """
    session = SessionLocal()
    try:
        run = (
            session.query(Run)
            .filter(Run.trace_id == trace_id)
            .order_by(Run.occurred_at.desc())
            .first()
        )

        if not run:
            return False

        run.user_feedback = user_feedback
        session.commit()
        
        # Record feedback metric
        record_runs_feedback(
            model_id=run.model_id,
            provider=run.provider_key,
            feedback=user_feedback,
        )
        
        return True
    except Exception as e:
        session.rollback()
        import logging

        logger = logging.getLogger(os.getenv("LOGGER_NAME", "llm-router"))
        logger.error(f"Failed to update user_feedback for trace_id={trace_id}: {e}", exc_info=True)
        return False
    finally:
        session.close()

