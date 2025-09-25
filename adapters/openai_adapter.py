import os, time, json
from typing import Generator, Optional, Literal
import openai
from utils.sse import sse_pack


def _unified_openai_api_call(model: str,
               messages,
               params: dict,
               stream: bool,
               api_key: str,
               base_url: str,
               stream_mode: Literal["raw","sse"]="raw",
               sse_event: str = "message",
               send_done: bool = False,              # optional status event
               include_metrics: bool = True          # <— add a final metrics event
               ):
    """
    Chat completion via OpenAI-compatible API.

    Works with OpenAI and any OpenAI-compatible server (e.g., vLLM) by
    optionally providing base_url.

    Args:
        model (str): Model name to use
        messages: Message content or list of messages
        params (dict): temperature, top_p, max_tokens, seed
        stream (bool): If True, returns streaming response generator
        api_key (str): API key
        base_url (str|None): Optional OpenAI-compatible base URL
        stream_mode: Stream in as SSE or plain text
        - 'raw' -> yields str chunks
        - 'sse' -> yields SSE bytes (text/event-stream ready), ends with metrics
        include_metrics: add TTFT and latency to SSE response

    Returns:
        If stream=True: Generator yielding streaming text chunks
        If stream=False: String with the complete response
    """
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    normalized_messages = (
        [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
    )
    kwargs = {
        "model": model,
        "messages": normalized_messages,
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "max_tokens": params.get("max_tokens"),
        "seed": params.get("seed"),
        "stream": stream,
    }
    if "gemini" in model:
        kwargs.pop("seed", None)

    t0 = time.perf_counter()
    res = client.chat.completions.create(**kwargs)

    def _iter_text(chunks):
        for ch in chunks:
            delta = ch.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece

    if stream:
        if stream_mode == "raw":
            # raw text stream with metrics appended as a final JSON line (optional)
            def _iter_raw() -> Generator[str, None, None]:
                ttft = None
                chunk_count = 0
                for piece in _iter_text(res):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunk_count += 1
                    yield piece
                total = time.perf_counter() - t0
                if include_metrics:
                    # emit a newline + JSON metrics so curl can capture it
                    metrics = {
                        "metrics": {
                            "ttft_ms": round(ttft * 1000, 1) if ttft is not None else None,
                            "total_ms": round(total * 1000, 1),
                            "chunks": chunk_count,
                            "model": model,
                        }
                    }
                    yield "\n" + json.dumps(metrics) + "\n"
            return _iter_raw()

        # SSE mode with a final `event: metrics`
        def _iter_sse() -> Generator[bytes, None, None]:
            ttft = None
            chunk_count = 0
            for piece in _iter_text(res):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunk_count += 1
                yield sse_pack(piece, event=sse_event)
            total = time.perf_counter() - t0
            if send_done:
                yield sse_pack("done", event="status")
            if include_metrics:
                payload = {
                    "ttft_ms": round(ttft * 1000, 1) if ttft is not None else None,
                    "total_ms": round(total * 1000, 1),
                    "chunks": chunk_count,
                    "model": model,
                }
                yield sse_pack(json.dumps(payload), event="metrics")  # <— final metrics event
        return _iter_sse()

    # non-streaming
    msg = res.choices[0].message
    return msg.content or ""   



def prettify_openai_error(err: Exception) -> str:
    """
    Convert OpenAI (or OpenAI-compatible) exceptions into human-readable messages.
    Uses HTTP code mapping + JSON 'error.message' if available.
    """
    # Mapping table: HTTP status codes → friendly message
    ERROR_MESSAGES = {
        400: "Bad Request: The request was malformed or contained invalid parameters.",
        401: "Authentication Error: API key is missing, invalid, or incorrect.",
        403: "Permission Denied: Your API key does not have access to this resource.",
        404: "Not Found: The requested resource (e.g., model) does not exist.",
        409: "Conflict: The request could not be processed because of a conflict.",
        422: "Unprocessable Entity: The request JSON was valid, but the input was invalid.",
        429: "Rate Limit Exceeded: Too many requests or quota exhausted. Retry later.",
        500: "Internal Server Error: Something went wrong on the server side.",
        503: "Service Unavailable: The server is overloaded or under maintenance.",
    }
    status_code = getattr(err, "status_code", None)
    resp = getattr(err, "response", None)

    # Handle network/timeout errors (no HTTP response)
    if isinstance(err, openai.APIConnectionError):
        return "Network Error: Could not reach the API (connection issue)."
    if isinstance(err, openai.APITimeoutError):
        return "Timeout Error: The API did not respond in time."

    # Base message from dictionary
    base_msg = str(status_code)+ ' | ' +ERROR_MESSAGES.get(status_code, f"Unknown error (status={status_code})")

    # Try to extract vendor-specific detail from response body
    detail = ""
    if resp is not None:
        try:
            data = resp.json()
            # OpenAI and compatible servers often wrap errors like {'error': {...}}
            if isinstance(data, dict) and "error" in data:
                if isinstance(data["error"], dict) and "message" in data["error"]:
                    detail = data["error"]["message"]
        except Exception:
            try:
                detail = resp.text[:200]
            except Exception:
                detail = str(err)[:200]

    if detail:
        return f"{base_msg} | Details: {detail}"
    return base_msg


