import json
import openai

def _unified_openai_api_call(model: str,
               messages,
               params: dict,
               stream: bool,
               api_key: str,
               base_url: str):
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

    Returns:
        If stream=True: Generator yielding streaming text chunks
        If stream=False: String with the complete response
    """
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    normalized_messages = (
    [{"role": "user", "content": messages}] if isinstance(messages, str) else messages)

    kwargs = {
        "model": model,
        "messages": normalized_messages,
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "max_tokens": params.get("max_tokens"),
        "seed": params.get("seed"),
        "stream": stream,
    }

    if 'gemini' in model:
        kwargs.pop('seed')

    res = client.chat.completions.create(**kwargs)

    if stream:
        def _streamer():
            for chunk in res:
                yield chunk.choices[0].delta.content or ""
        return _streamer()

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


