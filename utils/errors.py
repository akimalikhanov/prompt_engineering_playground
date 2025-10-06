import openai
from typing import Optional

class BackendError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# --- Expandable, human-facing messages ---
ERROR_MESSAGES = {
    400: "Bad Request: The request was malformed or contained invalid parameters.",
    401: "Authentication Error: API key is missing, invalid, or incorrect.",
    403: "Permission Denied: Your API key does not have access to this resource.",
    404: "Not Found: The requested resource (e.g., model) does not exist.",
    409: "Conflict: The request could not be processed because of a conflict.",
    413: "Payload Too Large: The request body is too big for the server.",
    415: "Unsupported Media Type: The server cannot process the content type.",
    422: "Unprocessable Entity: The JSON was valid, but the input was invalid.",
    408: "Request Timeout: The server timed out waiting for the request.",
    429: "Rate Limit Exceeded: Too many requests or quota exhausted. Retry later.",
    500: "Internal Server Error: Something went wrong on the server side.",
    502: "Bad Gateway: An upstream server failed. This may be temporary.",
    503: "Service Unavailable: The server is overloaded or under maintenance.",
    504: "Gateway Timeout: The upstream server timed out.",
}

# Minimal transient set for OpenAI-compatible chat APIs
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}

def _get_status_code(err: Exception) -> Optional[int]:
    return getattr(err, "status_code", None) or getattr(getattr(err, "response", None), "status_code", None)

def _get_response(err: Exception):
    return getattr(err, "response", None)

def _get_retry_after_seconds(err: Exception) -> Optional[float]:
    """Honor Retry-After header if present (seconds or HTTP-date; we only handle seconds)."""
    resp = _get_response(err)
    try:
        if resp is not None and hasattr(resp, "headers"):
            ra = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
            if not ra:
                return None
            # best-effort parse: prefer integer seconds
            return float(ra)
    except Exception:
        pass
    return None

def is_network_or_timeout(err: Exception) -> bool:
    return isinstance(err, (openai.APIConnectionError, openai.APITimeoutError))

def is_rate_limit(err: Exception) -> bool:
    # Works for OpenAI & most compatibles
    return isinstance(err, getattr(openai, "RateLimitError", Exception)) or _get_status_code(err) == 429

def is_transient_error(err: Exception) -> bool:
    """Classify transient: network/timeout, 429, or known 5xx/edge timeouts."""
    if is_network_or_timeout(err) or is_rate_limit(err):
        return True
    sc = _get_status_code(err)
    return sc in TRANSIENT_STATUS_CODES

def prettify_openai_error(err: Exception) -> str:
    """
    Convert OpenAI (or OpenAI-compatible) exceptions into human-readable messages.
    Uses HTTP code mapping + JSON 'error.message' if available.
    """
    status_code = _get_status_code(err)
    resp = _get_response(err)

    # Handle network/timeout errors (no HTTP response)
    if isinstance(err, openai.APIConnectionError):
        return "Network Error: Could not reach the API (connection issue)."
    if isinstance(err, openai.APITimeoutError):
        return "Timeout Error: The API did not respond in time."
    if is_rate_limit(err) and status_code is None:
        status_code = 429

    base_msg = f"{status_code} | " + ERROR_MESSAGES.get(
        status_code, f"Unknown error (status={status_code})"
    )

    # Try to extract vendor-specific detail from response body
    detail = ""
    if resp is not None:
        # Try JSON → {'error': {'message': '...'}}
        try:
            data = resp.json()
            if isinstance(data, dict) and "error" in data:
                err_block = data["error"]
                if isinstance(err_block, dict):
                    # Prefer message; fall back to 'type' or 'code'
                    if "message" in err_block:
                        detail = str(err_block["message"])
                    elif "type" in err_block:
                        detail = f"type={err_block['type']}"
                    elif "code" in err_block:
                        detail = f"code={err_block['code']}"
        except Exception:
            # Fall back to raw text snippet
            try:
                detail = resp.text[:200]
            except Exception:
                detail = str(err)[:200]

    if detail:
        return f"{base_msg} | Details: {detail}"
    return base_msg