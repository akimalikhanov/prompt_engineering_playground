import logging
from typing import Any, Dict, List, Optional

import gradio as gr
import httpx

from utils.http import resolve_api_base


def _send_feedback_to_backend(
    trace_id: Optional[str],
    user_feedback: int,
    api_base_url: Optional[str],
    default_api_base_url: str,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    POST feedback to the backend feedback endpoint.
    Returns True if request succeeded, False otherwise.
    """
    if not trace_id:
        return False

    feedback_url = None
    log = logger or logging.getLogger(__name__)

    try:
        api_url = resolve_api_base(api_base_url, default_api_base_url)
        feedback_url = f"{api_url.rstrip('/')}/runs/feedback"
        payload = {"trace_id": trace_id, "user_feedback": user_feedback}
        with httpx.Client(timeout=10.0) as client:
            response = client.post(feedback_url, json=payload)
            response.raise_for_status()
        return True
    except Exception as exc:
        log.error(
            "frontend_feedback_error",
            extra={
                "event": "frontend_feedback_error",
                "trace_id": trace_id,
                "session_id": None,
                "request_type": "POST",
                "api_endpoint": feedback_url or "unknown",
                "backend_endpoint": "/runs/feedback",
                "status": "error",
                "http_status": getattr(getattr(exc, "response", None), "status_code", None),
                "error_msg": str(exc),
            },
        )
        return False


def _apply_feedback_to_message(
    history: List[Dict[str, Any]],
    index: Optional[int],
    liked_state: Optional[bool],
) -> List[Dict[str, Any]]:
    """
    Apply user_feedback to a specific assistant message (by index) in the chat history.
    This only updates local UI state; backend persistence will be handled separately.
    """
    if not history:
        return history
    if index is None or index < 0 or index >= len(history):
        return history

    msg = history[index] or {}
    if msg.get("role") != "assistant":
        return history

    content = str(msg.get("content", ""))
    if content.startswith("🔧"):
        return history

    if liked_state is True:
        feedback_value = 1
    elif liked_state is False:
        feedback_value = -1
    else:
        feedback_value = 0

    msg["user_feedback"] = feedback_value
    return history


def _on_like_event(
    evt: "gr.LikeData",
    history: List[Dict[str, Any]],
    api_base_url: Optional[str],
    session_state: Optional[Dict[str, Any]],
    *,
    default_api_base_url: str,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Gradio Chatbot.like callback.
    evt.liked:
      - True  -> thumbs up
      - False -> thumbs down
      - None  -> reaction cleared
    """
    history = history or []
    index = getattr(evt, "index", None)
    liked_state = getattr(evt, "liked", None)

    history = _apply_feedback_to_message(history, index, liked_state)

    user_feedback = 0
    if (
        index is not None
        and 0 <= index < len(history)
        and isinstance(history[index], dict)
    ):
        user_feedback = history[index].get("user_feedback", 0)

    trace_id = None
    if isinstance(session_state, dict):
        trace_map = session_state.get("trace_map") or {}
        trace_id = trace_map.get(index)

    _send_feedback_to_backend(
        trace_id,
        user_feedback,
        api_base_url,
        default_api_base_url,
        logger,
    )
    return history


