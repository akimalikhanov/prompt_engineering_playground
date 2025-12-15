import json
import os
import time
from functools import partial
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
import httpx
import logging

from services.llm_tools import get_all_tool_schemas
from ui.styles import BADGES_CSS, TEXTAREA_SCROLL_CSS, ENHANCED_UI_CSS, NOTE_CSS
from utils.delimiter_utils import RESOLVER_FUNCTIONS, _delim_pair, _insert_pair_at_end, _wrap_entire_message
from utils.errors import _parse_error_payload, _parse_error_text, _format_api_error_message
from utils.feedback_utils import _on_like_event
from utils.http import resolve_api_base
from utils.load_configs import (
    _get_default_endpoint_key,
    _load_delimiter_definitions,
    _load_endpoints,
    _load_models,
    _load_response_schema_templates,
)
from utils.logger_new import setup_logging
from utils.message_utils import (
    _build_messages,
    _build_response_format,
    _coerce_seed,
    _normalize_selection,
    split_rendered_messages_for_ui,
)
from utils.model_params_utils import (
    _get_model_special_behavior,
    _get_param_values_for_model,
    get_effective_params_for_model,
)
from utils.prompts_client import get_prompt, list_prompts, render_prompt
from utils.render_formatters import format_prompt_details, format_render_preview, format_render_status
from utils.response_schema_utils import build_schema_from_template, extract_response_format_from_prompt
from utils.session_utils import (
    INACTIVITY_TIMEOUT,
    _ensure_session_state,
    _record_trace_id_from_metrics,
    _update_session_from_metrics,
)
from utils.ui_formatters import format_metrics_badges
from utils.otel_config import configure_otel_env_vars


MODELS_DATA = _load_models()
MODEL_LOOKUP = {model["id"]: model for model in MODELS_DATA["models"]}
DEFAULT_MODEL_ID = "gemini-flash-lite" if "gemini-flash-lite" in MODEL_LOOKUP else next(iter(MODEL_LOOKUP.keys()), None)

if DEFAULT_MODEL_ID is None:
    raise RuntimeError("No models configured in config/models.yaml")

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

UI_LOGGER_NAME = os.getenv("UI_LOGGER_NAME", "ui")
UI_OTEL_LOGS_ENABLED = os.getenv("OTEL_LOGS_ENABLED", "true").lower() == "true"

# Structured JSON logger for the UI process (console only; backend handles LLM metrics).
# Set OTEL env vars early so OTLP exporter has service/resource metadata.
configure_otel_env_vars()
ui_logger = setup_logging(
    app_logger_name=UI_LOGGER_NAME,
    level=logging.INFO,
    to_console=True,
    to_file=False,
    hijack_uvicorn=True,
    enable_otel=UI_OTEL_LOGS_ENABLED,
)

ENDPOINTS = _load_endpoints()
DEFAULT_ENDPOINT_KEY = _get_default_endpoint_key()

# Load delimiter and schema template configurations
DELIMITER_DEFINITIONS: Dict[str, Dict[str, Any]] = _load_delimiter_definitions()
DELIMITER_CHOICES: List[str] = list(DELIMITER_DEFINITIONS.keys())

RESPONSE_SCHEMA_TEMPLATES: Dict[str, Dict[str, Any]] = _load_response_schema_templates()
RESPONSE_SCHEMA_CHOICES: List[str] = list(RESPONSE_SCHEMA_TEMPLATES.keys())

DEFAULT_PROMPT_KEY = "bullet-summary"
TOOL_PROMPT_KEYS = {
    "movie-details-tool",
    "arxiv-search-tool",
    "company-fundamentals-tool",
}
STATEMENTS_VARIABLE = "statements"
STATEMENT_OPTIONS = ["income", "balance"]
STATEMENT_OPTION_SET = set(STATEMENT_OPTIONS)


def stream_chat(
    user_message: str,
    history: List[Dict[str, str]],
    model_choice: Any,
    endpoint_choice: Any,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed_input: Optional[str],
    reasoning_effort: Optional[str],
    verbosity: Optional[str],
    system_prompt: Optional[str],
    context_prompt: Optional[str],
    response_mode: str,
    strict_schema: bool,
    schema_code: Optional[str],
    schema_template: Optional[str] = None,
    enable_tools: bool = False,
    session_state: Optional[Dict[str, Any]] = None,
) -> Generator[Tuple[List[Dict[str, str]], str, Dict[str, Any]], None, None]:
    """Stream chat responses. Always yields values, even for errors.
    
    Returns a 3-tuple on each yield:
        - updated chat history
        - metrics markdown
        - updated session_state dict (for gr.State)
    """
    history = history or []
    # Sanitize history to only allowed fields and ensure string content
    sanitized_history: List[Dict[str, str]] = []
    for m in history:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role is None or content is None:
            continue
        # Coerce to strings
        role_str = str(role)
        content_str = str(content)
        sanitized_history.append({"role": role_str, "content": content_str})
    
    # Ensure we have a valid session and enforce 15-minute inactivity timeout
    session_id, session_state_out, started_new_session, session_expired = _ensure_session_state(session_state)
    if started_new_session:
        # New session -> reset chat history for this conversation
        if session_expired:
            # Only notify user if session actually expired (not first time visit)
            session_expired_msg = {
                "role": "assistant",
                "content": f"⚠️ **Session expired**: Your previous session timed out after {INACTIVITY_TIMEOUT} minutes of inactivity. Chat history has been reset. Starting a new conversation."
            }
            history = [session_expired_msg]
        else:
            # First time visit - just clear history without notification
            history = []
        sanitized_history = []

    # Validate response_format early and show error in UI if needed
    if response_mode == "JSON schema":
        if not schema_code or not schema_code.strip():
            error_msg = "⚠️ JSON schema mode requires a schema. Please enter a valid JSON schema."
            error_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": error_msg}]
            yield error_history, format_metrics_badges({}, None, None), session_state_out
            return
        try:
            # Validate JSON before proceeding
            parsed_schema = json.loads(schema_code.strip())
            # Also validate it's a dict
            if not isinstance(parsed_schema, dict):
                error_msg = f"⚠️ JSON schema must be an object/dict, got {type(parsed_schema).__name__}."
                error_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": error_msg}]
                yield error_history, format_metrics_badges({}, None, None), session_state_out
                return
        except json.JSONDecodeError as e:
            error_msg = f"⚠️ Invalid JSON schema: {str(e)}\n\nPlease check your JSON syntax (missing brackets, commas, etc.)."
            error_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": error_msg}]
            yield error_history, format_metrics_badges({}, None, None), session_state_out
            return

    normalized_choice = _normalize_selection(model_choice)
    model_config = MODEL_LOOKUP.get(normalized_choice) if normalized_choice else None
    if not model_config:
        error_msg = "⚠️ Unknown model selection."
        error_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": error_msg}]
        yield error_history, format_metrics_badges({}, None, None), session_state_out
        return

    # Handle tools: if enabled, force non-streaming endpoint
    # if enable_tools:
    #    endpoint_key = "chat"  # Force non-streaming endpoint
    #    endpoint_cfg = ENDPOINTS.get(endpoint_key)
    #    if not endpoint_cfg:
    #        error_msg = "⚠️ Non-streaming endpoint not found. Tools require non-streaming mode."
    #        error_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": error_msg}]
    #        yield error_history, format_metrics_badges({}, None, None), session_state_out
    #        return
    # else:
    endpoint_key = _normalize_selection(endpoint_choice)
    endpoint_cfg = ENDPOINTS.get(endpoint_key) if endpoint_key else None
    if not endpoint_cfg:
        error_msg = "⚠️ Unknown endpoint selection."
        error_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": error_msg}]
        yield error_history, format_metrics_badges({}, None, None), session_state_out
        return

    provider_id = model_config["provider"]
    provider_key = provider_id.lower() if isinstance(provider_id, str) else provider_id
    # Gemini (Google) does not support tool calling and structured output simultaneously
    disable_structured_output = enable_tools and provider_key == "google"

    payload_messages = _build_messages(user_message, sanitized_history, system_prompt, context_prompt)

    api_url = DEFAULT_API_BASE_URL.strip()
    if not api_url:
        api_url = DEFAULT_API_BASE_URL
    endpoint_url = f"{api_url.rstrip('/')}{endpoint_cfg['path']}"

    params: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": int(max_tokens),
    }

    seed_value = _coerce_seed(seed_input)
    if seed_value is not None:
        params["seed"] = seed_value

    # Add reasoning_effort and verbosity for GPT-5 models
    if reasoning_effort and reasoning_effort.strip():
        params["reasoning_effort"] = reasoning_effort
    if verbosity and verbosity.strip():
        params["verbosity"] = verbosity

    # Handle response_format (skip for Gemini when tools are enabled)
    schema_template_label = _normalize_selection(schema_template) if schema_template else None
    response_format = None
    if not disable_structured_output:
        response_format = _build_response_format(response_mode, strict_schema, schema_code, schema_template_label, RESPONSE_SCHEMA_TEMPLATES)
    if response_format is not None:
        params["response_format"] = response_format

    # Handle tools if enabled
    if enable_tools:
        try:
            tools = get_all_tool_schemas()
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"
        except Exception as e:
            error_msg = f"⚠️ Error loading tools: {str(e)}"
            error_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": error_msg}]
            yield error_history, format_metrics_badges({}, None, None), session_state_out
            return

    request_body = {
        "provider_id": provider_id,
        "model_id": model_config["id"],
        "messages": payload_messages,
        "params": params,
        "context_prompt": context_prompt.strip() if context_prompt and context_prompt.strip() else None,
        # Note: backend ChatRequest will be extended to accept this field
        "session_id": session_id,
    }

    # Add current user message to history
    running_history = history + [{"role": "user", "content": user_message}]
    ttft_ms: Optional[float] = None
    metrics: Dict[str, Any] = {}
    start_time = time.perf_counter()

    # Emit initial state to display user message immediately.
    yield running_history, format_metrics_badges(metrics, ttft_ms, None), session_state_out

    accumulated_response = ""
    timeout = httpx.Timeout(60.0, connect=10.0, read=None, write=10.0)

    try:
        with httpx.Client(timeout=timeout) as client:
            if endpoint_cfg["stream"]:
                with client.stream("POST", endpoint_url, json=request_body) as response:
                    response.raise_for_status()
                    if endpoint_cfg["sse"]:
                        done = False
                        sse_buffer = ""
                        for chunk in response.iter_bytes():
                            if done:
                                break
                            if not chunk:
                                continue
                            sse_buffer += chunk.decode("utf-8")
                            while "\n\n" in sse_buffer and not done:
                                event, sse_buffer = sse_buffer.split("\n\n", 1)
                                event_str = event.lstrip()  # Only strip leading whitespace to preserve trailing spaces in data
                                if not event_str:
                                    continue
                                lowered = event_str.lower()
                                if "event: done" in lowered or "data: [done]" in lowered:
                                    done = True
                                    break
                                if "event: error" in event_str:
                                    error_status = None
                                    error_detail = None
                                    for line in event_str.split("\n"):
                                        if line.startswith("data: "):
                                            status, message = _parse_error_text(line[6:].strip())
                                            if status is not None:
                                                error_status = status
                                            if message:
                                                error_detail = message
                                    error_msg = _format_api_error_message(error_status, error_detail)
                                    running_history.append({"role": "assistant", "content": error_msg})
                                    yield running_history, format_metrics_badges(metrics, ttft_ms, None), session_state_out
                                    return
                                if "event: metrics" in event_str:
                                    for line in event_str.split("\n"):
                                        if line.startswith("data: "):
                                            try:
                                                metrics_data = json.loads(line[6:])
                                                metrics.clear()
                                                metrics.update(metrics_data)
                                                # Update session_state if session_id is in metrics
                                                session_state_out = _update_session_from_metrics(metrics, session_state_out)
                                                # Attach trace_id (if present) to the main assistant message
                                                running_history = _record_trace_id_from_metrics(
                                                    running_history,
                                                    metrics,
                                                    session_state_out,
                                                    logger=ui_logger,
                                                    source="sse_metrics_event",
                                                )
                                            except json.JSONDecodeError:
                                                pass
                                    continue
                                if "event: tool_status" in event_str:
                                    # Extract tool status message
                                    for line in event_str.split("\n"):
                                        if line.startswith("data: "):
                                            tool_msg = line[6:].strip()
                                            if tool_msg:
                                                # Add tool execution message to history
                                                running_history.append({"role": "assistant", "content": tool_msg})
                                                yield running_history, format_metrics_badges(metrics, ttft_ms, None), session_state_out
                                    continue
                                if "event: message" in event_str:
                                    data_lines = []
                                    for line in event_str.split("\n"):
                                        if line.startswith("data:"):
                                            # Extract content after "data:", handling optional space per SSE spec
                                            v = line[5:]
                                            if v.startswith(" "):
                                                v = v[1:]
                                            data_lines.append(v)
                                    new_text = "\n".join(data_lines)
                                    if not new_text:
                                        continue
                                    if ttft_ms is None:
                                        ttft_ms = (time.perf_counter() - start_time) * 1000.0
                                    accumulated_response += new_text
                                    # Update assistant response in messages format
                                    # If last message is user (first chunk), append new assistant message
                                    # If last message is a tool_status message, append new assistant message to preserve tool message
                                    # Otherwise (subsequent chunks), update existing assistant message
                                    last_msg = running_history[-1] if running_history else {}
                                    is_tool_status = last_msg.get("content", "").startswith("🔧")
                                    if running_history[-1].get("role") == "user" or is_tool_status:
                                        running_history.append({"role": "assistant", "content": accumulated_response})
                                    else:
                                        # Preserve any metadata (trace_id, feedback state) when updating content
                                        if isinstance(running_history[-1], dict):
                                            running_history[-1]["content"] = accumulated_response
                                        else:
                                            running_history[-1] = {"role": "assistant", "content": accumulated_response}
                                    latency_now = (time.perf_counter() - start_time) * 1000.0
                                    yield running_history, format_metrics_badges(metrics, ttft_ms, latency_now), session_state_out
                        # Process any remaining buffer content (incomplete final event)
                        if not done and sse_buffer.strip():
                            event_str = sse_buffer.lstrip()  # Only strip leading whitespace to preserve trailing spaces in data
                            # Process metrics events
                            if "event: error" in event_str:
                                error_status = None
                                error_detail = None
                                for line in event_str.split("\n"):
                                    if line.startswith("data: "):
                                        status, message = _parse_error_text(line[6:].strip())
                                        if status is not None:
                                            error_status = status
                                        if message:
                                            error_detail = message
                                error_msg = _format_api_error_message(error_status, error_detail)
                                running_history.append({"role": "assistant", "content": error_msg})
                                yield running_history, format_metrics_badges(metrics, ttft_ms, None), session_state_out
                                return
                            if "event: metrics" in event_str:
                                for line in event_str.split("\n"):
                                    if line.startswith("data: "):
                                        try:
                                            metrics_data = json.loads(line[6:])
                                            metrics.clear()
                                            metrics.update(metrics_data)
                                            # Update session_state if session_id is in metrics
                                            session_state_out = _update_session_from_metrics(metrics, session_state_out)
                                            # Attach trace_id (if present) to the main assistant message
                                            running_history = _record_trace_id_from_metrics(
                                                running_history,
                                                metrics,
                                                session_state_out,
                                                logger=ui_logger,
                                                source="sse_final_buffer",
                                            )
                                        except json.JSONDecodeError:
                                            pass
                            # Process tool_status events
                            elif "event: tool_status" in event_str:
                                for line in event_str.split("\n"):
                                    if line.startswith("data: "):
                                        tool_msg = line[6:].strip()
                                        if tool_msg:
                                            running_history.append({"role": "assistant", "content": tool_msg})
                                            yield running_history, format_metrics_badges(metrics, ttft_ms, None), session_state_out
                            # Also process message events in the final buffer
                            elif "event: message" in event_str:
                                data_lines = []
                                for line in event_str.split("\n"):
                                    if line.startswith("data:"):
                                        # Extract content after "data:", handling optional space per SSE spec
                                        v = line[5:]
                                        if v.startswith(" "):
                                            v = v[1:]
                                        data_lines.append(v)
                                new_text = "\n".join(data_lines)
                                if new_text:
                                    if ttft_ms is None:
                                        ttft_ms = (time.perf_counter() - start_time) * 1000.0
                                    
                                    # Check if this is a tool execution log
                                    # Tool logs usually start with "🔧" or contain specific markers
                                    # But in our backend, tool logs are currently just part of the text stream or separate messages
                                    # We need to append them to history carefully
                                    
                                    accumulated_response += new_text
                                    
                                    # Update assistant response in messages format
                                    # If last message is user (first chunk), append new assistant message
                                    # If last message is a tool_status message, append new assistant message to preserve tool message
                                    # Otherwise (subsequent chunks), update existing assistant message
                                    last_msg = running_history[-1] if running_history else {}
                                    is_tool_status = last_msg.get("content", "").startswith("🔧")
                                    if running_history[-1].get("role") == "user" or is_tool_status:
                                        running_history.append({"role": "assistant", "content": accumulated_response})
                                    else:
                                        if isinstance(running_history[-1], dict):
                                            running_history[-1]["content"] = accumulated_response
                                        else:
                                            running_history[-1] = {"role": "assistant", "content": accumulated_response}
                                    latency_now = (time.perf_counter() - start_time) * 1000.0
                                    yield running_history, format_metrics_badges(metrics, ttft_ms, latency_now), session_state_out
                    else:
                        for chunk in response.iter_text():
                            if not chunk:
                                continue
                            json_payload = None
                            if chunk.startswith("\n{"):
                                try:
                                    json_payload = json.loads(chunk.strip())
                                except json.JSONDecodeError:
                                    pass
                            if isinstance(json_payload, dict):
                                if "error" in json_payload:
                                    status, detail = _parse_error_payload(json_payload)
                                    error_msg = _format_api_error_message(status, detail)
                                    running_history.append({"role": "assistant", "content": error_msg})
                                    yield running_history, format_metrics_badges(metrics, ttft_ms, None), session_state_out
                                    return
                                if "metrics" in json_payload:
                                    metrics_obj = json_payload.get("metrics", json_payload)
                                    metrics = metrics_obj
                                    session_state_out = _update_session_from_metrics(metrics, session_state_out)
                                    running_history = _record_trace_id_from_metrics(
                                        running_history,
                                        metrics,
                                        session_state_out,
                                        logger=ui_logger,
                                        source="raw_stream_metrics",
                                    )
                                    continue
                                if "tool_status" in json_payload:
                                    tool_msg = json_payload.get("tool_status", "")
                                    if tool_msg:
                                        running_history.append({"role": "assistant", "content": tool_msg})
                                        yield running_history, format_metrics_badges(metrics, ttft_ms, None), session_state_out
                                    continue
                                continue
                            
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - start_time) * 1000.0

                            accumulated_response += chunk
                            # Update assistant response in messages format
                            # If last message is user (first chunk), append new assistant message
                            # If last message is a tool_status message, append new assistant message to preserve tool message
                            # Otherwise (subsequent chunks), update existing assistant message
                            last_msg = running_history[-1] if running_history else {}
                            is_tool_status = last_msg.get("content", "").startswith("🔧")
                            if running_history[-1].get("role") == "user" or is_tool_status:
                                running_history.append({"role": "assistant", "content": accumulated_response})
                            else:
                                if isinstance(running_history[-1], dict):
                                    running_history[-1]["content"] = accumulated_response
                                else:
                                    running_history[-1] = {"role": "assistant", "content": accumulated_response}
                            latency_now = (time.perf_counter() - start_time) * 1000.0
                            yield running_history, format_metrics_badges(metrics, ttft_ms, latency_now), session_state_out
            else:
                # Non-streaming endpoint
                response = client.post(endpoint_url, json=request_body)
                response.raise_for_status()
                data = response.json()
                
                # Extract text, metrics, and tool messages from response
                text = data.get("text")
                metrics = data.get("metrics", {})
                tool_messages = data.get("tool_messages", [])  # Backend-formatted tool messages
                
                # Add tool execution messages before the assistant response (from backend)
                if tool_messages:
                    for tool_msg in tool_messages:
                        running_history.append({
                            "role": "assistant",
                            "content": tool_msg
                        })
                
                # Handle text response - always use text if it exists, even if empty
                if text is not None:
                    running_history.append({"role": "assistant", "content": str(text) if text else "⚠️ Empty response body."})
                else:
                    running_history.append({"role": "assistant", "content": "⚠️ Empty response body."})

                # Warn if reasoning tokens likely consumed the budget leaving no visible output
                if metrics.get("reasoning_tokens", 0) > 0:
                    content = running_history[-1].get("content", "") if running_history else ""
                    if not str(content).strip():
                        warning_msg = (
                            "⚠️ All available tokens were used for reasoning, so no visible answer was returned. "
                            "Try increasing max tokens or lowering reasoning_effort."
                        )
                        warning_entry = {"role": "assistant", "content": warning_msg, "likeable": False}
                        if running_history and running_history[-1].get("role") == "assistant":
                            running_history[-1] = warning_entry
                        else:
                            running_history.append(warning_entry)
                
                # Update session_state if session_id is in metrics
                session_state_out = _update_session_from_metrics(metrics, session_state_out)
                # Attach trace_id (if present) to the main assistant message
                running_history = _record_trace_id_from_metrics(
                    running_history,
                    metrics,
                    session_state_out,
                    logger=ui_logger,
                    source="non_stream_response",
                )

                if ttft_ms is None and metrics.get("ttft_ms") is not None:
                    ttft_ms = metrics["ttft_ms"]
                final_latency = (time.perf_counter() - start_time) * 1000.0
                yield running_history, format_metrics_badges(metrics, ttft_ms, final_latency), session_state_out
                return

    except httpx.HTTPStatusError as http_err:
        response = http_err.response
        detail_text = ""
        parsed_status = None
        try:
            content_bytes = response.read()
            if content_bytes:
                detail_text = content_bytes.decode("utf-8", errors="ignore").strip()
        except Exception:
            detail_text = ""
        if not detail_text:
            detail_text = getattr(response, "reason_phrase", "") or str(http_err)
        else:
            parsed_status, parsed_detail = _parse_error_text(detail_text)
            if parsed_detail:
                detail_text = parsed_detail
        status_for_display = parsed_status or response.status_code
        error_msg = _format_api_error_message(status_for_display, detail_text)

        # Structured UI log for backend HTTP errors
        ui_logger.error(
            "frontend_http_error",
            extra={
                "event": "frontend_http_error",
                "trace_id": metrics.get("trace_id"),
                "session_id": session_id,
                "provider": provider_id,
                "model_id": model_config["id"],
                "request_type": "POST",
                "api_endpoint": endpoint_url,
                "backend_endpoint": endpoint_cfg["path"],
                "http_status": status_for_display,
                "status": "error",
                "error_msg": detail_text,
            },
        )

        running_history.append({"role": "assistant", "content": error_msg})
        yield running_history, format_metrics_badges(metrics, ttft_ms, None), session_state_out
        return
    except httpx.HTTPError as transport_err:
        # Transport-level issues (DNS, connect timeout, etc.) that never reach backend
        ui_logger.error(
            "frontend_transport_error",
            extra={
                "event": "frontend_transport_error",
                "trace_id": metrics.get("trace_id"),
                "session_id": session_id,
                "provider": provider_id,
                "model_id": model_config["id"],
                "request_type": "POST",
                "api_endpoint": endpoint_url,
                "backend_endpoint": endpoint_cfg["path"],
                "status": "error",
                "error_msg": str(transport_err),
            },
        )
        error_msg = f"⚠️ Connection error: {transport_err}"
        running_history.append({"role": "assistant", "content": error_msg})
        yield running_history, format_metrics_badges(metrics, ttft_ms, None), session_state_out
        return
    except Exception as exc:
        ui_logger.error(
            "frontend_unexpected_error",
            extra={
                "event": "frontend_unexpected_error",
                "trace_id": metrics.get("trace_id"),
                "session_id": session_id,
                "provider": provider_id,
                "model_id": model_config["id"],
                "request_type": "POST",
                "api_endpoint": endpoint_url,
                "backend_endpoint": endpoint_cfg["path"],
                "status": "error",
                "error_msg": str(exc),
            },
        )
        running_history.append({"role": "assistant", "content": f"⚠️ Unexpected error: {exc}"})
        yield running_history, format_metrics_badges(metrics, ttft_ms, None), session_state_out
        return

    final_latency = (time.perf_counter() - start_time) * 1000.0
    # Ensure assistant response is in messages format
    # If last message is user (first chunk), append new assistant message
    # If last message is a tool_status message, append new assistant message to preserve tool message
    # Otherwise (subsequent chunks), update existing assistant message
    # If no visible response, insert a placeholder to keep history valid for retries
    if not (accumulated_response or "").strip():
        accumulated_response = "<no response>"

    if running_history:
        last_msg = running_history[-1]
        is_tool_status = last_msg.get("content", "").startswith("🔧")
        if running_history[-1].get("role") == "user" or is_tool_status:
            running_history.append({"role": "assistant", "content": accumulated_response})
        else:
            if isinstance(running_history[-1], dict):
                running_history[-1]["content"] = accumulated_response
            else:
                running_history[-1] = {"role": "assistant", "content": accumulated_response}
    else:
        running_history.append({"role": "assistant", "content": accumulated_response})

    # Warn if reasoning tokens likely consumed the budget leaving no visible output
    if metrics.get("reasoning_tokens", 0) > 0 and not (accumulated_response or "").strip():
        warning_msg = (
            "⚠️ All available tokens were used for reasoning, so no visible answer was returned. "
            "Try increasing max tokens or lowering reasoning_effort."
        )
        warning_entry = {"role": "assistant", "content": warning_msg, "likeable": False}
        if running_history and running_history[-1].get("role") == "assistant":
            running_history[-1] = warning_entry
        else:
            running_history.append(warning_entry)
    yield running_history, format_metrics_badges(metrics, ttft_ms, final_latency), session_state_out




# --- Helpers (moved out of build_demo) ---------------------------------------

def sync_delimiter_inputs(selected_style: str):
    definition = DELIMITER_DEFINITIONS.get(selected_style or "")
    if not definition:
        return gr.update(visible=False), gr.update(visible=False)

    start_update = gr.update(
        visible=definition.get("requires_start", False),
        label=definition.get("start_label", "Custom start"),
        placeholder=definition.get("start_placeholder", "e.g. [[[INSTRUCT]]]"),
    )
    end_update = gr.update(
        visible=definition.get("requires_end", False),
        label=definition.get("end_label", "Custom end"),
        placeholder=definition.get("end_placeholder", "e.g. [[[/INSTRUCT]]]"),
    )
    return start_update, end_update


def handle_insert_end(text, style, cs, ce):
    return _insert_pair_at_end(text, style, cs, ce, DELIMITER_DEFINITIONS)


def handle_wrap_all(text, style, cs, ce):
    return _wrap_entire_message(text, style, cs, ce, DELIMITER_DEFINITIONS)


def clear_chat_and_metrics():
    """Clear chat, metrics, and reset key input fields to defaults."""
    return [], format_metrics_badges({}, None, None), "", "", "", "None"


def on_schema_template_change(template_label: str, current_strict: bool) -> Tuple[str, bool]:
    """Handle schema template selection via utils."""
    return build_schema_from_template(template_label, RESPONSE_SCHEMA_TEMPLATES, current_strict)


def reset_textbox():
    return ""


def update_params_for_model(model_choice: Any):
    """Update all parameter UI components based on selected model.
    
    This is a generic function that:
    1. Looks up the model's params_override from models.yaml
    2. Merges with defaults to get effective params
    3. Applies special behavior rules (e.g., fixed values, disabled UI)
    4. Returns updates for all parameter UI components
    
    Returns:
        Tuple of gr.update() calls for all parameter UI components in order:
        (temperature_slider, top_p_slider, max_tokens_slider, seed_textbox,
         reasoning_effort_dropdown, verbosity_dropdown)
    """
    normalized_choice = _normalize_selection(model_choice)
    params = _get_param_values_for_model(normalized_choice, MODELS_DATA, MODEL_LOOKUP)
    
    return (
        gr.update(**params["temperature"]),
        gr.update(value=params["top_p"]["value"]),
        gr.update(value=params["max_tokens"]["value"]),
        gr.update(value=params["seed"]["value"]),
        gr.update(
            value=params["reasoning_effort"]["value"],
            visible=params["reasoning_effort"]["visible"],
            choices=params["reasoning_effort"].get("choices"),
        ),
        gr.update(
            value=params["verbosity"]["value"],
            visible=params["verbosity"]["visible"],
        ),
    )


def _generation_help_text(model_choice: Any) -> str:
    """Return Generation tab help text, including GPT-5-only options when applicable."""
    normalized_choice = _normalize_selection(model_choice)
    is_gpt5 = normalized_choice.startswith("gpt-5") if normalized_choice else False

    base_text = (
        "### ⚙️ Generation\n"
        "Fine-tune how the model produces text:\n\n"
        "- **Temperature** – controls creativity (lower is more focused, higher is more diverse).\n"
        "- **Top‑p (nucleus)** – limits sampling to the most probable tokens for steadier output.\n"
        "- **Max tokens** – caps the response length.\n"
        "- **Seed** – optionally set to make results more reproducible with the same inputs and parameters.\n"
    )

    if is_gpt5:
        base_text += (
        "- **Reasoning effort** – controls how much deliberate thinking the model uses (higher improves logic but increases cost/latency).\n"
        "- **Verbosity** – controls how detailed the response is (low = concise, high = more explanation and context).\n"
        )

    return base_text


def fetch_prompts():
    """Fetch prompts from the API."""
    try:
        api_url = DEFAULT_API_BASE_URL
        prompts = list_prompts(api_url)
        
        # Create choices for dropdown: (label, prompt_id)
        # Use current schema field names (id, key) and allow technique_key for label fallback
        choices = []
        preferred_default = None
        fallback_default = None
        for p in prompts:
            prompt_id = p.get('id')
            prompt_key = p.get('key') or p.get('technique_key', 'unknown')
            prompt_title = p.get('title', 'Untitled')
            if prompt_id:
                choices.append((f"{prompt_title} ({prompt_key})", str(prompt_id)))
                str_prompt_id = str(prompt_id)
                if preferred_default is None and prompt_key == DEFAULT_PROMPT_KEY:
                    preferred_default = str_prompt_id
                if fallback_default is None and prompt_key not in TOOL_PROMPT_KEYS:
                    fallback_default = str_prompt_id
        
        default_value = preferred_default or fallback_default or (choices[0][1] if choices else None)
        return gr.update(choices=choices, value=default_value)
    except Exception as e:
        return gr.update(choices=[("Error loading prompts", None)], value=None)


def _prompt_requires_tools(prompt: Optional[Dict[str, Any]]) -> bool:
    """Return True if the prompt should auto-enable tool calling."""
    if not isinstance(prompt, dict):
        return False
    tool_config = prompt.get("tool_config")
    if tool_config:
        return True
    prompt_key = (prompt.get("key") or "").strip()
    return prompt_key in TOOL_PROMPT_KEYS


def _default_statements_selection(default_value: Optional[Any]) -> List[str]:
    """Return checkbox defaults for the statements selector based on template defaults."""
    if default_value is None or default_value == "":
        return STATEMENT_OPTIONS.copy()
    if isinstance(default_value, bool):
        return STATEMENT_OPTIONS.copy() if default_value else []
    if isinstance(default_value, (int, float)):
        return STATEMENT_OPTIONS.copy() if default_value else []
    if isinstance(default_value, (list, tuple, set)):
        return _normalize_statements_selection(list(default_value))
    if isinstance(default_value, str):
        stripped = default_value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return _normalize_statements_selection(parsed)
        except json.JSONDecodeError:
            pass
        parts = [part.strip() for part in stripped.split(",")]
        return _normalize_statements_selection([part for part in parts if part])
    return []


def _normalize_statements_selection(selection: Any) -> List[str]:
    """Normalize checkbox selections (list/str) into canonical lowercase values."""
    if selection is None:
        return []
    if isinstance(selection, str):
        candidates = [selection]
    elif isinstance(selection, (list, tuple, set)):
        candidates = selection
    else:
        return []
    normalized: List[str] = []
    for candidate in candidates:
        value = str(candidate).strip().lower()
        if value in STATEMENT_OPTION_SET and value not in normalized:
            normalized.append(value)
    return normalized



def paste_template_to_input(rendered_json: str, prompt_id: str = None):
    """Parse the rendered JSON and distribute messages to appropriate fields.
    
    - System messages -> system_prompt_box
    - User messages -> user_input (combined)
    - Other messages -> included in user_input with role markers
    - response_format and json_schema_template -> response_mode and schema_code
    """
    if not rendered_json or not rendered_json.strip():
        return "", "", "None", ""
    
    try:
        messages = json.loads(rendered_json)
        if not isinstance(messages, list):
            return "", "", "None", ""

        split = split_rendered_messages_for_ui(messages)
        system_prompt = split.get("system_text", "")
        user_input_text = split.get("user_text", "")

        # Fetch prompt details to get response_format and json_schema_template
        response_mode = "None"
        schema_code = ""
        
        if prompt_id:
            try:
                api_url = DEFAULT_API_BASE_URL
                prompt = get_prompt(api_url, prompt_id)
                response_mode, schema_code = extract_response_format_from_prompt(prompt)
            except Exception:
                # If we can't fetch prompt details, just use defaults
                pass
        return system_prompt, user_input_text, response_mode, schema_code
    except json.JSONDecodeError:
        # If it's not valid JSON, return empty
        return "", "", "None", ""
    except Exception as e:
        return "", "", "None", ""


def submit_chat_generator(*args):
    for value in stream_chat(*args):
        yield value


# --- Main UI builder ---------------------------------------------------------

def build_demo() -> gr.Blocks:
    models_choices = [
        (
            f"{model['label']} ({model['provider']})",
            model["id"],
        )
        for model in MODELS_DATA["models"]
    ]
    endpoint_choices = [
        (endpoint_cfg["label"], endpoint_key)
        for endpoint_key, endpoint_cfg in ENDPOINTS.items()
    ]
    defaults = MODELS_DATA["defaults"]

    with gr.Blocks(
        css=BADGES_CSS + TEXTAREA_SCROLL_CSS + ENHANCED_UI_CSS + NOTE_CSS,
        theme=gr.themes.Ocean(primary_hue="emerald"),
        title="Prompt Engineering Playground",
    ) as demo:
        gr.Markdown(
            "# 🧪 Prompt Engineering Playground\n"
            "Experiment with API & local LLMs, tune generation, design prompts & templates, and test tool-calling — all in one place."
        )

        with gr.Row():
            # --- Left: chat area ------------------------------------------------
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=520,
                    type="messages",
                    show_copy_button=True,
                    # avatar_images: [user, assistant]; None = no avatar
                    avatar_images=[
                        None,  # no user icon
                        "utils/icons/icons8-bot-96.png",  # assistant icon only
                    ],
                )
                user_input = gr.Textbox(
                    label="Your message",
                    placeholder="Ask something...",
                    autofocus=True,
                    lines=3,
                    elem_id="prompt-input",
                    elem_classes=["scroll-text"],
                )
                with gr.Row():
                    send_button = gr.Button(
                        "Send", variant="primary", elem_id="send-button"
                    )
                    clear_button = gr.Button("Clear", variant="secondary")

                metrics_display = gr.Markdown(
                    value=format_metrics_badges({}, None, None),
                    label="Metrics",
                )

                # Hidden state for session tracking (session_id + last_activity timestamp)
                session_state = gr.State(value=None)

            # --- Right: settings / parameters ----------------------------------
            with gr.Column(scale=2, min_width=420):
                with gr.Accordion("Parameters", open=True):
                    # General
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            label="Model",
                            choices=models_choices,
                            value=DEFAULT_MODEL_ID,
                        )
                        endpoint_dropdown = gr.Dropdown(
                            label="Endpoint",
                            choices=endpoint_choices,
                            value=DEFAULT_ENDPOINT_KEY,
                        )

                        # Wire the chatbot like/dislike control now that we know API base input.
                        # NOTE: In Gradio 5.x, inputs come first, then gr.LikeData is
                        # auto-injected based on the type hint (not position).
                        def _like_handler(
                            history: List[Dict[str, Any]],
                            session: Optional[Dict[str, Any]],
                            evt: gr.LikeData,
                        ) -> List[Dict[str, Any]]:
                            # Skip feedback only on the targeted message if it's a warning
                            idx = getattr(evt, "index", None)
                            if (
                                idx is not None
                                and 0 <= idx < len(history)
                                and isinstance(history[idx], dict)
                            ):
                                msg = history[idx]
                                content = str(msg.get("content") or "").strip()
                                likeable = msg.get("likeable")
                                if likeable is False or content.startswith("⚠️"):
                                    return history

                            return _on_like_event(
                                evt,
                                history,
                                DEFAULT_API_BASE_URL,
                                session,
                                default_api_base_url=DEFAULT_API_BASE_URL,
                                logger=ui_logger,
                            )

                        chatbot.like(
                            _like_handler,
                            inputs=[chatbot, session_state],
                            outputs=[chatbot],
                        )

                    # Tabs for logical grouping
                    with gr.Tabs():
                        # --- Generation tab ------------------------------------
                        with gr.Tab("Generation"):
                            generation_help_md = gr.Markdown(
                                _generation_help_text(DEFAULT_MODEL_ID)
                            )
                            with gr.Row():
                                # Initialize parameters based on default model (merge defaults with params_override)
                                initial_params = _get_param_values_for_model(DEFAULT_MODEL_ID, MODELS_DATA, MODEL_LOOKUP)
                                
                                temperature_slider = gr.Slider(
                                    label=initial_params["temperature"]["label"],
                                    minimum=0.0,
                                    maximum=2.0,
                                    step=0.1,
                                    value=initial_params["temperature"]["value"],
                                    interactive=initial_params["temperature"]["interactive"],
                                )
                                
                                top_p_slider = gr.Slider(
                                    label="Top-p",
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    value=initial_params["top_p"]["value"],
                                )
                            with gr.Row():
                                max_tokens_slider = gr.Slider(
                                    label="Max tokens",
                                    minimum=1,
                                    maximum=4000,
                                    step=1,
                                    value=initial_params["max_tokens"]["value"],
                                )
                                
                                seed_text = gr.Textbox(
                                    label="Seed (optional)",
                                    placeholder="e.g. 42",
                                    value=initial_params["seed"]["value"],
                                    elem_classes=["scroll-text"],
                                )

                            with gr.Row():
                                reasoning_effort_dd = gr.Dropdown(
                                    label="Reasoning effort (GPT-5 models only)",
                                    choices=initial_params["reasoning_effort"]["choices"],
                                    value=initial_params["reasoning_effort"]["value"],
                                    visible=initial_params["reasoning_effort"]["visible"],
                                )
                                
                                verbosity_dd = gr.Dropdown(
                                    label="Verbosity (GPT-5 models only)",
                                    choices=["low", "medium", "high"],
                                    value=initial_params["verbosity"]["value"],
                                    visible=initial_params["verbosity"]["visible"],
                                )

                            # --- Structured output (response_format) ------------
                            with gr.Accordion("Structured output (optional)", open=False):
                                gr.Markdown(
                                    "### 🧱 Structured output (JSON)\n"
                                    "Use `response_format` to produce machine-readable responses:\n\n"
                                    "- **None** – normal text response\n"
                                    "- **JSON object** – any valid JSON object (older option)\n"
                                    "- **JSON schema** – enforce a specific JSON structure (recommended, newer option)"
                                )

                                response_mode_dd = gr.Dropdown(
                                    label="Mode",
                                    choices=["None", "JSON object", "JSON schema"],
                                    value="None",
                                )

                                strict_ckb = gr.Checkbox(
                                    label="Strict schema (JSON only, no extra fields)",
                                    value=True,
                                    visible=False,
                                )

                                schema_template_dd = gr.Dropdown(
                                    label="Schema template",
                                    choices=RESPONSE_SCHEMA_CHOICES,
                                    value="Custom...",
                                    visible=False,
                                    interactive=True,
                                )

                                schema_code = gr.Code(
                                    label="JSON schema",
                                    language="json",
                                    value='{\n  "type": "object",\n  "additionalProperties": false,\n'
                                          '  "properties": {},\n  "required": []\n}',
                                    visible=False,
                                    lines=12,
                                    max_lines=20,
                                )

                                # Small change handler to show/hide fields when mode changes
                                def on_response_mode_change(mode):
                                    show_schema = mode == "JSON schema"
                                    return (
                                        gr.update(visible=show_schema),  # strict_ckb
                                        gr.update(visible=show_schema),  # schema_template_dd
                                        gr.update(visible=show_schema),  # schema_code
                                    )

                                response_mode_dd.change(
                                    on_response_mode_change,
                                    inputs=[response_mode_dd],
                                    outputs=[strict_ckb, schema_template_dd, schema_code],
                                    queue=False,
                                )

                                # Handler for schema template selection
                                def on_schema_template_select(template_label: str, current_strict: bool):
                                    """Handle template selection and update schema code and strict checkbox."""
                                    schema_json, strict_val = on_schema_template_change(template_label, current_strict)
                                    return (
                                        gr.update(value=schema_json),  # schema_code
                                        gr.update(value=strict_val),  # strict_ckb
                                    )

                                schema_template_dd.change(
                                    on_schema_template_select,
                                    inputs=[schema_template_dd, strict_ckb],
                                    outputs=[schema_code, strict_ckb],
                                    queue=False,
                                )

                        # --- Prompting tab -------------------------------------
                        with gr.Tab("Prompting"):
                            gr.Markdown(
                                "### 💬 Prompting\n"
                                "- **System prompt**: set high-level behavior, tone, or role for the assistant.\n"
                                "- **Context**: ground answers in your facts (docs, retrieved snippets, examples).\n\n"
                                "Use system prompt for **high-level behavior** and Context to **ground answers in facts**."
                            )
                            system_prompt_box = gr.Textbox(
                                label="System prompt (optional)",
                                placeholder="Add global instructions, tone, or role.",
                                lines=3,
                                elem_classes=["scroll-text"],
                            )

                            context_box = gr.Textbox(
                                label="Context (optional)",
                                placeholder=(
                                    "Add retrieved facts or reference material here. "
                                    "This will be sent as a special `CONTEXT:` user "
                                    "message before your main question."
                                ),
                                lines=3,
                                elem_classes=["scroll-text"],
                            )

                            with gr.Accordion("Delimiters (advanced)", open=False):
                                gr.Markdown(
                                    "### ✂️ Delimiters\n"
                                    "Delimiters create clear **boundaries** in prompts between instructions, context, and answers:\n\n"
                                    "- Help parsing structured segments.\n"
                                    "- Reduce prompt-injection risks.\n"
                                    "- Make templates easier to reuse across prompts.\n"
                                )

                                delim_pair_dd = gr.Dropdown(
                                    choices=DELIMITER_CHOICES,
                                    value=DELIMITER_CHOICES[0],
                                    label="Delimiter style",
                                    interactive=True,
                                )
                                with gr.Row():
                                    custom_start_tb = gr.Textbox(
                                        label="Custom start",
                                        placeholder="e.g. [[[INSTRUCT]]]",
                                        visible=False,
                                        elem_classes=["scroll-text"],
                                    )
                                    custom_end_tb = gr.Textbox(
                                        label="Custom end",
                                        placeholder="e.g. [[[/INSTRUCT]]]",
                                        visible=False,
                                        elem_classes=["scroll-text"],
                                    )

                                delim_pair_dd.change(
                                    sync_delimiter_inputs,
                                    inputs=[delim_pair_dd],
                                    outputs=[custom_start_tb, custom_end_tb],
                                    queue=False,
                                )

                                with gr.Row():
                                    insert_delim_btn = gr.Button(
                                        "Insert at end", variant="secondary"
                                    )
                                    wrap_all_btn = gr.Button(
                                        "Wrap all text", variant="secondary"
                                    )

                                insert_delim_btn.click(
                                    handle_insert_end,
                                    inputs=[
                                        user_input,
                                        delim_pair_dd,
                                        custom_start_tb,
                                        custom_end_tb,
                                    ],
                                    outputs=[user_input],
                                    queue=False,
                                )
                                wrap_all_btn.click(
                                    handle_wrap_all,
                                    inputs=[
                                        user_input,
                                        delim_pair_dd,
                                        custom_start_tb,
                                        custom_end_tb,
                                    ],
                                    outputs=[user_input],
                                    queue=False,
                                )

                        # --- Prompt Hub tab -------------------------------------
                        with gr.Tab("Prompt Hub"):
                            gr.Markdown(
                                "### 📚 Prompt Hub\n"
                                "Load and reuse prompt templates rendered with Jinja variables:\n\n"
                                "- Select a prompt from the list to see its details.\n"
                                "- **Simple mode**: fill variables in the textboxes (multiline supported).\n"
                                "- **Few-shot**: if available, click **Paste examples to Context** to add `[EXAMPLES]`.\n"
                                "- Click **Render Template** to preview, then **Paste template** to send to chat.\n"
                                "- **Advanced JSON**: toggle to edit the raw variables JSON directly."
                            )
                            
                            # Maximum number of variables to support (can be increased if needed)
                            MAX_VARIABLES = 6
                            
                            # Advanced/Simple toggle
                            advanced_mode_ckb = gr.Checkbox(
                                label="Advanced JSON",
                                value=False,
                                # elem_classes=["enhanced-ui-checkbox"],
                            )
                            
                            prompt_dropdown = gr.Dropdown(
                                label="Select Prompt",
                                choices=[],
                                value=None,
                                interactive=True,
                            )
                            
                            prompt_description = gr.Markdown(
                                value="Select a prompt to see its details.",
                                label="Prompt Details"
                            )
                            
                            variables_json = gr.Code(
                                label="Variables (JSON)",
                                language="json",
                                value="{}",
                                lines=8,
                                visible=False,  # hidden in Simple mode by default
                            )
                            
                            # --- Simple mode inputs (auto-built from variables) ---
                            with gr.Group(visible=True) as simple_mode_group:
                                # Dynamically create variable textboxes based on MAX_VARIABLES
                                var_name_hidden_list = []
                                var_value_tb_list = []
                                for i in range(MAX_VARIABLES):
                                    var_name_hidden_list.append(gr.Textbox(visible=False))
                                    var_value_tb_list.append(gr.Textbox(
                                        label=f"var_{i+1}",
                                        lines=3,
                                        visible=False,
                                        elem_classes=["scroll-text"]
                                    ))
                                statements_selector = gr.CheckboxGroup(
                                    label="Financial statements to include",
                                    choices=STATEMENT_OPTIONS,
                                    value=[],
                                    visible=False,
                                    info="Choose which statements get_fmp_company_data should request (income, balance). Leave empty for profile only.",
                                )
                            
                            with gr.Row():
                                render_btn = gr.Button(
                                    "Render Template", variant="primary"
                                )
                                paste_btn = gr.Button(
                                    "Paste template", variant="secondary"
                                )
                            # This button appears only for few_shot technique
                            paste_examples_btn = gr.Button(
                                "Paste examples to Context", variant="secondary", visible=False
                            )
                            
                            rendered_output = gr.Textbox(
                                label="Rendered Template",
                                lines=10,
                                interactive=False,
                                placeholder="Rendered template will appear here..."
                                ,
                                elem_classes=["scroll-text"],
                            )
                            
                            # Hidden component to store raw JSON for pasting
                            rendered_json_hidden = gr.Textbox(
                                visible=False,
                                elem_classes=["scroll-text"],
                            )
                            # Hidden component to store formatted examples text
                            examples_text_hidden = gr.Textbox(
                                visible=False,
                                elem_classes=["scroll-text"],
                            )
                            
                            render_status = gr.Markdown(
                                value="",
                                label="Render Status"
                            )
                            
                            # --- Functions for Simple/Advanced behavior -----------
                            def load_prompt_details_and_simple_fields(prompt_id: str):
                                """Load prompt details, JSON defaults, and configure simple-mode fields."""
                                if not prompt_id:
                                    # Reset UI to defaults - clear all variable fields
                                    reset_updates = [
                                        gr.update(value="Select a prompt to see its details."),
                                        gr.update(value="{}"),
                                        gr.update(value=""),
                                    ]
                                    # Clear all variable fields
                                    for i in range(MAX_VARIABLES):
                                        reset_updates.extend([
                                            gr.update(value=""),
                                            gr.update(visible=False, label=f"var_{i+1}", value="")
                                        ])
                                    reset_updates.append(gr.update(visible=False, value=[]))  # statements selector
                                    reset_updates.append(gr.update(visible=False))  # paste_examples_btn visibility
                                    return tuple(reset_updates)
                                
                                try:
                                    api_url = DEFAULT_API_BASE_URL
                                    prompt = get_prompt(api_url, prompt_id)
                                    description_text, defaults_json = format_prompt_details(prompt)
                                    # Build examples text for context
                                    examples = prompt.get("default_examples")
                                    examples_text = ""
                                    if examples:
                                        try:
                                            examples_text = "[EXAMPLES]\n\n" + json.dumps(examples, indent=2)
                                        except Exception:
                                            examples_text = "[EXAMPLES]\n\n" + str(examples)
                                    # Determine technique for button visibility
                                    technique_val = (prompt.get("technique") or prompt.get("technique_key") or "").strip().lower()
                                    show_examples_btn = technique_val == "few_shot"
                                    # Build variable fields strictly from template variables
                                    variables = prompt.get("variables", []) or []
                                    num_vars = len(variables)
                                    statements_default_value: Optional[Any] = None
                                    statements_field_present = False
                                    
                                    # Helper to build updates for one slot
                                    def slot_updates(var_def):
                                        nonlocal statements_default_value, statements_field_present
                                        if var_def:
                                            label = var_def.get("name") or "variable"
                                            default_val = var_def.get("default", "")
                                            hide_statements_flag = (label == STATEMENTS_VARIABLE)
                                            if hide_statements_flag:
                                                statements_field_present = True
                                                statements_default_value = default_val
                                            return (
                                                gr.update(value=label),
                                                gr.update(
                                                    visible=not hide_statements_flag,
                                                    label=label,
                                                    value=default_val,
                                                ),
                                            )
                                        else:
                                            return gr.update(value=""), gr.update(visible=False, label="var", value="")
                                    
                                    # Build updates for all slots
                                    updates = [
                                        gr.update(value=description_text),
                                        gr.update(value=defaults_json),
                                        gr.update(value=examples_text),
                                    ]
                                    
                                    # Update visible slots based on actual variables
                                    for i in range(MAX_VARIABLES):
                                        var_def = variables[i] if i < num_vars else None
                                        updates.extend(slot_updates(var_def))
                                    
                                    statements_visible = statements_field_present
                                    statements_default = (
                                        _default_statements_selection(statements_default_value)
                                        if statements_visible else []
                                    )
                                    updates.append(gr.update(visible=statements_visible, value=statements_default))
                                    updates.append(gr.update(visible=show_examples_btn))
                                    return tuple(updates)
                                except Exception as e:
                                    error_msg = f"Error loading prompt: {str(e)}"
                                    error_updates = [
                                        gr.update(value=error_msg),
                                        gr.update(value="{}"),
                                        gr.update(value=""),
                                    ]
                                    # Clear all variable fields on error
                                    for i in range(MAX_VARIABLES):
                                        error_updates.extend([
                                            gr.update(value=""),
                                            gr.update(visible=False, label=f"var_{i+1}", value="")
                                        ])
                                    error_updates.append(gr.update(visible=False, value=[]))
                                    error_updates.append(gr.update(visible=False))
                                    return tuple(error_updates)
                            
                            def toggle_advanced_mode(is_advanced: bool):
                                """Show/hide simple inputs vs raw JSON editor."""
                                return (
                                    gr.update(visible=is_advanced),   # variables_json
                                    gr.update(visible=not is_advanced),  # simple_mode_group
                                )
                            
                            def render_template_with_modes(
                                prompt_id: str,
                                is_advanced: bool,
                                variables_json_str: str,
                                context_text: str,
                                *var_inputs,  # Accept variable name/value pairs + statements selector
                            ):
                                """Render, building variables from simple fields if not advanced."""
                                try:
                                    # Last argument: statements selector value
                                    statements_selection_raw: Any = []
                                    if var_inputs:
                                        statements_selection_raw = var_inputs[-1]
                                        var_inputs = var_inputs[:-1]
                                    statements_selection = (
                                        _normalize_statements_selection(statements_selection_raw)
                                        if not is_advanced else []
                                    )
                                    
                                    api_url = DEFAULT_API_BASE_URL
                                    if is_advanced:
                                        # Use JSON as-is
                                        try:
                                            variables = json.loads(variables_json_str) if variables_json_str.strip() else {}
                                        except json.JSONDecodeError as e:
                                            return "", "", f"⚠️ Invalid JSON in variables: {str(e)}"
                                    else:
                                        # Collect variables from all textboxes dynamically
                                        # var_inputs contains: vn1, vv1, vn2, vv2, ..., vnN, vvN (all pairs)
                                        variables = {}
                                        # Process pairs: each pair is (name, value)
                                        num_pairs = len(var_inputs) // 2
                                        for i in range(num_pairs):
                                            var_name = var_inputs[i * 2] if i * 2 < len(var_inputs) else ""
                                            var_value = var_inputs[i * 2 + 1] if i * 2 + 1 < len(var_inputs) else ""
                                            if (var_name or "").strip():
                                                variables[var_name.strip()] = var_value if var_value is not None else ""
                                    statements_enabled = False
                                    if not is_advanced and STATEMENTS_VARIABLE in variables:
                                        statements_enabled = True
                                        variables[STATEMENTS_VARIABLE] = statements_selection
                                    # Call API
                                    data = render_prompt(api_url, prompt_id, variables)
                                    rendered_messages = data.get("rendered_messages", [])
                                    if not rendered_messages:
                                        return "", "", "⚠️ No rendered messages returned."
                                    # Build ordered preview: SYSTEM, EXAMPLES (if present in context), then others (USER, etc.)
                                    system_blocks = []
                                    other_blocks = []
                                    for msg in rendered_messages:
                                        role = (msg.get("role") or "unknown").lower()
                                        content = msg.get("content", "")
                                        if role == "system":
                                            system_blocks.append(f"[SYSTEM]\n{content}")
                                        else:
                                            other_blocks.append(f"[{role.upper()}]\n{content}")
                                    parts = []
                                    if system_blocks:
                                        parts.append("\n\n".join(system_blocks))
                                    # Only show EXAMPLES if user pasted them into Context
                                    ctx = (context_text or "").strip()
                                    if ctx.startswith("[EXAMPLES]"):
                                        parts.append(ctx)
                                    if other_blocks:
                                        parts.append("\n\n".join(other_blocks))
                                    rendered_display = "\n\n".join(parts) if parts else ""
                                    rendered_json = json.dumps(rendered_messages, indent=2)
                                    missing_vars = data.get("missing_vars", [])
                                    warnings = data.get("warnings", [])
                                    status_msg = format_render_status(missing_vars, warnings)
                                    return rendered_display, rendered_json, status_msg
                                except httpx.HTTPStatusError as e:
                                    error_detail = ""
                                    try:
                                        error_detail = e.response.json().get("detail", str(e))
                                    except:
                                        error_detail = str(e)
                                    return "", "", f"⚠️ API error: {error_detail}"
                                except Exception as e:
                                    return "", "", f"⚠️ Error rendering template: {str(e)}"
                            
                            def paste_examples_to_context(examples_text: str, _current_context: str):
                                """Override Context with formatted examples under [EXAMPLES] header."""
                                ex = (examples_text or "").strip()
                                return ex
                            
                            # Wire up the events
                            # Build outputs list dynamically for prompt_dropdown.change
                            prompt_outputs = [
                                prompt_description, variables_json, examples_text_hidden,
                            ]
                            # Add all variable textbox pairs
                            for i in range(MAX_VARIABLES):
                                prompt_outputs.extend([var_name_hidden_list[i], var_value_tb_list[i]])
                            prompt_outputs.append(statements_selector)
                            prompt_outputs.append(paste_examples_btn)
                            
                            prompt_dropdown.change(
                                load_prompt_details_and_simple_fields,
                                inputs=[prompt_dropdown],
                                outputs=prompt_outputs,
                                queue=False,
                            )
                            
                            advanced_mode_ckb.change(
                                toggle_advanced_mode,
                                inputs=[advanced_mode_ckb],
                                outputs=[variables_json, simple_mode_group],
                                queue=False,
                            )
                            
                            # Build inputs list dynamically for render_btn.click
                            render_inputs = [
                                prompt_dropdown,
                                advanced_mode_ckb,
                                variables_json,
                                context_box,
                            ]
                            # Add all variable name/value pairs
                            for i in range(MAX_VARIABLES):
                                render_inputs.extend([var_name_hidden_list[i], var_value_tb_list[i]])
                            render_inputs.append(statements_selector)
                            
                            render_btn.click(
                                render_template_with_modes,
                                inputs=render_inputs,
                                outputs=[rendered_output, rendered_json_hidden, render_status],
                                queue=False,
                            )
                            
                            paste_btn.click(
                                paste_template_to_input,
                                inputs=[rendered_json_hidden, prompt_dropdown],
                                outputs=[system_prompt_box, user_input, response_mode_dd, schema_code],
                                queue=False,
                            )
                            
                            paste_examples_btn.click(
                                paste_examples_to_context,
                                inputs=[examples_text_hidden, context_box],
                                outputs=[context_box],
                                queue=False,
                            )

                        # --- Tool Calling tab -------------------------------------
                        with gr.Tab("Tool Calling"):
                            def get_tools_info():
                                """Get formatted info about available tools as bullet list."""
                                try:
                                    tools = get_all_tool_schemas()
                                    if not tools:
                                        return "No tools available."
                                    
                                    tool_items = []
                                    for tool in tools:
                                        func_info = tool.get("function", {})
                                        name = func_info.get("name", "Unknown")
                                        desc = func_info.get("description", "No description available.")
                                        tool_items.append(f"- **{name}**: {desc}")
                                    
                                    return "\n".join(tool_items)
                                except Exception as e:
                                    return f"⚠️ Error loading tools: {str(e)}"
                            
                            gr.Markdown(
                                "### 🛠️ Tool Calling\n"
                                "Enable the model to call external tools/functions for real-time data and actions:\n\n"
                                "- When **Enable tools** is checked, the model can automatically call available tools when helpful.\n"
                                "- Tools are supported in both streaming and non-streaming modes.\n"
                                "- The system prompt is enhanced to instruct the model to use tools when necessary.\n"
                            )
                            
                            enable_tools_ckb = gr.Checkbox(
                                label="Enable tools",
                                value=False,
                                elem_classes=["enhanced-ui-checkbox"],
                            )
                            
                            tool_prompt_override_state = gr.State(value=False)
                            
                            gr.Markdown("#### 🔌 Available tools")
                            
                            # Display available tools with descriptions as bullet list
                            available_tools_md = gr.Markdown(
                                value=get_tools_info(),
                            )
                            
                            gr.Markdown(
                                "ℹ️ **Notes:**\n"
                                '''
                                - `get_fmp_company_data` supports income and balance sheet data for a limited number of companies. For more details, see [FMP API documentation](https://site.financialmodelingprep.com/developer/docs).
                                - Gemini does not currently support using tool calling and structured output in the same request. (see [issue](https://github.com/llm-router/llm-router/issues/706)). To avoid errors, the system automatically switches the response format to plain text when Gemini + tools are both enabled. 
                                If you need structured output + tools, consider either using OpenAI models or making two calls (tools → text, then text→ structured output).
                                ''',
                                elem_classes=["note-style"],
                            )

                with gr.Row():
                    gr.Markdown(
                        value=f"ℹ️ **Note**: Your conversation will expire after {INACTIVITY_TIMEOUT} minutes of inactivity.",
                            label=None,
                            elem_classes=["session-info", "note-style"],  # Optional: for custom styling
                        )

        # --- Wiring submissions -----------------------------------------------
        submit_inputs = [
            user_input,
            chatbot,
            model_dropdown,
            endpoint_dropdown,
            temperature_slider,
            top_p_slider,
            max_tokens_slider,
            seed_text,
            reasoning_effort_dd,
            verbosity_dd,
            system_prompt_box,
            context_box,
            response_mode_dd,
            strict_ckb,
            schema_code,
            schema_template_dd,
            enable_tools_ckb,
            session_state,
        ]
        submit_outputs = [chatbot, metrics_display, session_state]

        user_input.submit(stream_chat, inputs=submit_inputs, outputs=submit_outputs, queue=True)
        user_input.submit(reset_textbox, inputs=None, outputs=user_input, queue=False)

        send_button.click(
            submit_chat_generator,
            inputs=submit_inputs,
            outputs=submit_outputs,
            queue=True,
        )
        send_button.click(reset_textbox, inputs=None, outputs=user_input, queue=False)

        clear_button.click(
            clear_chat_and_metrics,
            inputs=None,
            outputs=[chatbot, metrics_display, user_input, system_prompt_box, context_box, response_mode_dd],
            queue=False,
        )

        # Update all parameter UI components when model changes
        model_dropdown.change(
            update_params_for_model,
            inputs=[model_dropdown],
            outputs=[temperature_slider, top_p_slider, max_tokens_slider, seed_text, reasoning_effort_dd, verbosity_dd],
            queue=False,
        )

        model_dropdown.change(
            lambda m: gr.update(value=_generation_help_text(m)),
            inputs=[model_dropdown],
            outputs=[generation_help_md],
            queue=False,
        )

        # Handle tools checkbox: disable/enable endpoint dropdown, force non-streaming, and set system prompt
        def on_tools_toggle(enable_tools: bool, current_system_prompt: str, allow_default_prompt: bool = True):
            """Handle Enable tools toggle and optionally insert the default tools system prompt."""
            tools_system_prompt = (
                "You are a helpful assistant with access to tools that can help you provide accurate and up-to-date information. "
                "When a user asks a question that can be answered using available tools, you should use them to get the most current and accurate information. "
                "After using tools, summarize the results clearly for the user."
            )
            
            if enable_tools:
                endpoint_update = gr.update(interactive=True)
                # Only set system prompt if allowed and it's currently empty
                if allow_default_prompt and (not current_system_prompt or not current_system_prompt.strip()):
                    system_prompt_update = gr.update(value=tools_system_prompt)
                else:
                    # Don't overwrite existing system prompt
                    system_prompt_update = gr.update()
                return endpoint_update, system_prompt_update
            else:
                # Re-enable endpoint selection, don't modify system prompt
                return gr.update(interactive=True), gr.update()

        def sync_tools_with_prompt_selection(prompt_id: str, current_system_prompt: str):
            """Auto-enable tools for prompts that declare tool usage."""
            if not prompt_id:
                return gr.update(), gr.update(), gr.update(), False
            try:
                api_url = DEFAULT_API_BASE_URL
                prompt = get_prompt(api_url, prompt_id)
            except Exception:
                return gr.update(), gr.update(), gr.update(), False
            
            if not _prompt_requires_tools(prompt):
                return gr.update(), gr.update(), gr.update(), False
            
            endpoint_update, system_prompt_update = on_tools_toggle(
                True,
                current_system_prompt,
                allow_default_prompt=False,
            )
            return gr.update(value=True), endpoint_update, system_prompt_update, True

        def handle_enable_tools_checkbox(enable_tools: bool, current_system_prompt: str, skip_default_prompt: bool):
            """Wrapper that respects one-shot skips for the default tools system prompt."""
            endpoint_update, system_prompt_update = on_tools_toggle(
                enable_tools,
                current_system_prompt,
                allow_default_prompt=not skip_default_prompt,
            )
            # Always reset skip flag after applying it
            return endpoint_update, system_prompt_update, False
        
        enable_tools_ckb.change(
            handle_enable_tools_checkbox,
            inputs=[enable_tools_ckb, system_prompt_box, tool_prompt_override_state],
            outputs=[endpoint_dropdown, system_prompt_box, tool_prompt_override_state],
            queue=False,
        )
        
        prompt_dropdown.change(
            sync_tools_with_prompt_selection,
            inputs=[prompt_dropdown, system_prompt_box],
            outputs=[enable_tools_ckb, endpoint_dropdown, system_prompt_box, tool_prompt_override_state],
            queue=False,
        )

        # Automatically load prompts when the demo loads
        demo.load(
            fetch_prompts,
            inputs=[],
            outputs=[prompt_dropdown],
            queue=False,
        )

        demo.queue()

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

