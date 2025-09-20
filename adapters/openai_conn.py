from openai import OpenAI
from openai import RateLimitError, AuthenticationError, APIConnectionError, APIError

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

    try:
        res = client.chat.completions.create(**kwargs)

        if stream:
            # inner generator so the outer function isn't a generator in non-streaming mode
            def _streamer():
                try:
                    for chunk in res:
                        yield chunk.choices[0].delta.content or ""
                except Exception as e:
                    raise _handle_openai_error(e)
            return _streamer()

        # non-streaming: return plain text
        msg = res.choices[0].message
        return msg.content or ""
    except Exception as e:
        raise _handle_openai_error(e)


def _handle_openai_error(error):
    """
    Convert OpenAI API errors to user-friendly messages.
    
    Args:
        error: The original exception from OpenAI API
        
    Returns:
        Exception: User-friendly exception with appropriate message
    """
    # Handle specific OpenAI exception types
    if isinstance(error, RateLimitError):
        return Exception(
            "🚫 Rate limit exceeded: You've hit your OpenAI API rate limit. "
            "Please wait a moment before trying again, or consider upgrading your plan."
        )
    elif isinstance(error, AuthenticationError):
        return Exception(
            "🔑 Authentication failed: Invalid or missing OpenAI API key. "
            "Please check your OPENAI_API_KEY environment variable."
        )
    elif isinstance(error, APIConnectionError):
        return Exception(
            "🌐 Connection failed: Unable to connect to OpenAI API. "
            "Please check your internet connection and try again."
        )
    elif isinstance(error, APIError):
        error_str = str(error)
        if "insufficient_quota" in error_str.lower():
            return Exception(
                "💰 Insufficient quota: You've exceeded your OpenAI API usage quota. "
                "Please check your billing and usage limits."
            )
        elif "model_not_found" in error_str.lower():
            return Exception(
                "🤖 Model not found: The specified OpenAI model is not available. "
                "Please check the model name in your configuration."
            )
        else:
            return Exception(f"🤖 OpenAI API error: {error_str}")
    
    # Generic error handling
    else:
        error_str = str(error)
        return Exception(
            f"🤖 OpenAI API error: {error_str[:200]}{'...' if len(error_str) > 200 else ''}"
        )
