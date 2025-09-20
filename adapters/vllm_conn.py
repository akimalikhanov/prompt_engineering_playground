from openai import OpenAI
from openai import RateLimitError, AuthenticationError, APIConnectionError, APIError

# vllm serve ./storage/llama3.1-8b-awq --port 8000 --max-model-len 8k --gpu-memory-utilization 0.8

def vllm_call(model: str,
               messages: str,
               params: dict,
               stream: bool,
               base_url: str):
    """
    Chat completion with vLLM server.
    
    Args:
        model (str): Model name/path to use
        messages (str): Message content.
        params (dict): Dictionary containing parameters like temperature, top_p, max_tokens, seed
        stream (bool): If True, returns streaming response. If False, returns complete response.
        base_url (str): Base URL for the vLLM server
    
    Returns:
        If stream=True: None (prints streaming response to console)
        If stream=False: None (prints complete response to console)
    
    Example:
        params = {"temperature": 0.7, "max_tokens": 1000}
        vllm_chat("./storage/llama3.1-8b-awq", "Hello!", params, stream=True)
    """
    client = OpenAI(
        base_url=base_url,
        api_key="dummy"  # vLLM ignores this, but the client requires it
    )
    
    # Handle messages - if it's a string, convert to list format
    messages = [{"role": "user", "content": messages}]
    
    # Build kwargs with parameters
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "max_tokens": params.get("max_tokens"),
        "seed": params.get("seed"),
        "stream": stream,
    }
    
    try:
        # Single resp definition with stream parameter
        res = client.chat.completions.create(**kwargs)
        
        if stream:
            # inner generator so the outer function isn't a generator in non-streaming mode
            def _streamer():
                try:
                    for chunk in res:
                        yield chunk.choices[0].delta.content or ""
                except Exception as e:
                    raise _handle_vllm_error(e)
            return _streamer()

        # non-streaming: return plain text
        msg = res.choices[0].message
        return msg.content or ""
    except Exception as e:
        raise _handle_vllm_error(e)


def _handle_vllm_error(error):
    """
    Convert vLLM server errors to user-friendly messages.
    
    Args:
        error: The original exception from vLLM server
        
    Returns:
        Exception: User-friendly exception with appropriate message
    """
    error_str = str(error)
    
    # Handle connection errors
    if "Connection refused" in error_str or "ConnectionError" in error_str:
        return Exception(
            "🔌 Connection failed: Unable to connect to vLLM server. "
            "Please ensure the vLLM server is running and check the server configuration."
        )
    elif "timeout" in error_str.lower():
        return Exception(
            "⏱️ Timeout: The vLLM server took too long to respond. "
            "The model might be overloaded or the request is too complex."
        )
    elif "out of memory" in error_str.lower() or "oom" in error_str.lower():
        return Exception(
            "💾 Out of memory: The vLLM server doesn't have enough GPU memory. "
            "Try reducing max_tokens or using a smaller model."
        )
    elif "model not found" in error_str.lower():
        return Exception(
            "🤖 Model not found: The specified model is not available on the vLLM server. "
            "Please check the model path in your configuration."
        )
    elif "rate limit" in error_str.lower():
        return Exception(
            "🚫 Rate limit exceeded: Too many requests to the vLLM server. "
            "Please wait a moment before trying again."
        )
    elif "authentication" in error_str.lower() or "unauthorized" in error_str.lower():
        return Exception(
            "🔑 Authentication failed: Invalid credentials for vLLM server. "
            "Please check your server configuration."
        )
    
    # Generic error handling
    else:
        return Exception(
            f"🤖 vLLM server error: {error_str[:200]}{'...' if len(error_str) > 200 else ''}"
        )


