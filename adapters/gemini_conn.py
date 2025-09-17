from google import genai

def gemini_call(model: str, 
                messages, 
                params: dict, 
                stream: bool, 
                api_key: str):
    """
    Chat completion with Google Gemini API.
    
    Args:
        model (str): Model name to use (e.g., "gemini-1.5-pro", "gemini-1.5-flash")
        messages: Message content or list of messages
        params (dict): Dictionary containing parameters like temperature, top_p, max_tokens, seed
        stream (bool): If True, returns streaming response. If False, returns complete response.
        api_key (str): Google Gemini API key for authentication
    
    Returns:
        If stream=True: Generator yielding streaming response chunks
        If stream=False: String containing the complete response
    
    Example:
        params = {"temperature": 0.7, "max_tokens": 1000}
        response = gemini_call("gemini-1.5-pro", "Hello!", params, stream=False, api_key="your-key")
    """
    # Initialize client
    client = genai.Client(api_key=api_key)
    
    cfg_kwargs = {
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "max_output_tokens": params.get("max_tokens"),
        "seed": params.get("seed"),
    }

    if stream:
        # Streaming mode - return generator like other connectors
        response = client.models.generate_content_stream(
            model=model,
            contents=messages,
            config=cfg_kwargs
        )
        # inner generator so the outer function isn't a generator in non-streaming mode
        def _streamer():
            for chunk in response:
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
        return _streamer()
    else:
        # Non-streaming mode
        response = client.models.generate_content(
            model=model,
            contents=messages,
            config=cfg_kwargs
        )
        return response.text
