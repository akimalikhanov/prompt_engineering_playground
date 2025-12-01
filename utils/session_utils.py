import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import os

INACTIVITY_TIMEOUT = int(os.getenv("INACTIVITY_TIMEOUT", 15))  # minutes
SESSION_TIMEOUT_SECONDS = INACTIVITY_TIMEOUT * 60


def _ensure_session_state(
    session_state: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], bool, bool]:
    """
    Ensure there is a valid session state and enforce inactivity timeout.

    Returns:
        session_id: active session identifier
        new_state: updated state dict to store in gr.State
        started_new_session: True if a new session was created (id changed or first time)
        session_expired: True if a previous session expired (vs. first time visit)
    """
    now = time.time()
    state = dict(session_state) if isinstance(session_state, dict) else {}
    session_id = state.get("session_id")
    last_activity = state.get("last_activity")

    started_new_session = False
    session_expired = False

    if (
        not session_id
        or not isinstance(last_activity, (int, float))
        or (now - last_activity) > SESSION_TIMEOUT_SECONDS
    ):
        if session_id and isinstance(last_activity, (int, float)):
            session_expired = True
        session_id = str(uuid.uuid4())
        last_activity = now
        started_new_session = True
        state = {}
    else:
        last_activity = now

    state["session_id"] = session_id
    state["last_activity"] = last_activity
    state.setdefault("trace_map", {})
    new_state = state
    return session_id, new_state, started_new_session, session_expired


def _update_session_from_metrics(
    metrics: Dict[str, Any],
    session_state_out: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Update session_state from session_id (and related tracking fields) in metrics if present.
    """
    if metrics.get("session_id"):
        session_state_out["session_id"] = metrics["session_id"]
        session_state_out["last_activity"] = time.time()
    return session_state_out


def _record_trace_id_from_metrics(
    history: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    session_state_out: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Record the trace_id from metrics into session_state's trace_map for the latest
    assistant message. We return history unchanged for backwards compatibility.
    """
    trace_id = metrics.get("trace_id")
    if not trace_id or not history:
        return history

    if session_state_out is None:
        return history

    trace_map = session_state_out.setdefault("trace_map", {})

    for idx in range(len(history) - 1, -1, -1):
        msg = history[idx]
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content", ""))
        if content.startswith("🔧"):
            continue
        trace_map[idx] = trace_id
        break

    return history


