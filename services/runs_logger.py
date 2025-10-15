"""
Service for logging LLM runs to app.runs table
"""
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.runs import Run

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

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
            # Concatenate all message contents
            texts = []
            for msg in messages:
                if isinstance(msg, dict) and 'content' in msg:
                    texts.append(f"{msg.get('role', 'user')}: {msg['content']}")
            combined = " | ".join(texts)
            return combined[:max_length]
    except Exception:
        pass
    return None


def log_run(
    trace_id: str,
    provider_key: str,
    model_id: str,
    messages: Any,
    params: Dict[str, Any],
    output_text: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
    latency_ms: Optional[int] = None,
    ttft_ms: Optional[int] = None,
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
        output_text: Model output text
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total tokens used
        cost_usd: Cost in USD
        latency_ms: Total latency in milliseconds
        ttft_ms: Time to first token in milliseconds
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
        metrics: Dict containing metrics (prompt_tokens, completion_tokens, total_tokens, 
                 cost_usd, latency_ms, ttft_ms). Values from this dict are used if 
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
            output_text=output_text[:2000] if output_text else None,  # Truncate for preview
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            status=status,
            error_type=error_type,
            error_code=error_code,
            error_message=error_message[:500] if error_message else None,
            pricing_snapshot=pricing_snapshot or {},
            metadata_json=metadata or {},
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

