from openai import OpenAI

def openai_call(model: str, 
                messages, 
                params: dict, 
                stream: bool, 
                api_key: str):
    """
    Chat completion with OpenAI API.
    
    Args:
        model (str): Model name to use (e.g., "gpt-4", "gpt-3.5-turbo")
        messages: Message content or list of messages
        params (dict): Dictionary containing parameters like temperature, top_p, max_tokens, seed
        stream (bool): If True, returns streaming response. If False, returns complete response.
        api_key (str): OpenAI API key for authentication
    
    Returns:
        If stream=True: Generator yielding streaming response chunks
        If stream=False: String containing the complete response
    
    Example:
        params = {"temperature": 0.7, "max_tokens": 1000}
        response = openai_call("gpt-4", "Hello!", params, stream=False, api_key="your-key")
    """
    client = OpenAI(api_key=api_key)
    
    messages = [{"role": "user", "content": messages}]

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "max_tokens": params.get("max_tokens"),
        "seed": params.get("seed"),
        "stream": stream,
    }

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
