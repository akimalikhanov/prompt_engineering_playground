from openai import OpenAI

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
    
    # Single resp definition with stream parameter
    res = client.chat.completions.create(**kwargs)
    
    if stream:
        # inner generator so the outer function isn't a generator in non-streaming mode
        def _streamer():
            for chunk in res:
                yield chunk.choices[0].delta.content or ""
        return _streamer()

    # non-streaming: return plain text
    msg = res.choices[0].message
    return msg.content or ""


