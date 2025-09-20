from google import genai
import time

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

    try:
        if stream:
            # Streaming mode - return generator like other connectors
            response = client.models.generate_content_stream(
                model=model,
                contents=messages,
                config=cfg_kwargs
            )
            # inner generator so the outer function isn't a generator in non-streaming mode
            def _streamer():
                try:
                    for chunk in response:
                        if hasattr(chunk, "text") and chunk.text:
                            yield chunk.text
                except Exception as e:
                    raise _handle_gemini_error(e)
            return _streamer()
        else:
            # Non-streaming mode
            response = client.models.generate_content(
                model=model,
                contents=messages,
                config=cfg_kwargs
            )
            return response.text
    except Exception as e:
        raise _handle_gemini_error(e)


def _handle_gemini_error(error):
    """
    Convert Gemini API errors to user-friendly messages.
    
    Args:
        error: The original exception from Gemini API
        
    Returns:
        Exception: User-friendly exception with appropriate message
    """
    error_str = str(error)
    
    # Handle rate limit errors
    if "429" in error_str and "RESOURCE_EXHAUSTED" in error_str:
        if "quota" in error_str.lower():
            return Exception(
                "🚫 Rate limit exceeded: You've hit your Gemini API quota limit. "
                "The free tier allows 15 requests per minute. "
                "Please wait a minute before trying again, or consider upgrading your plan."
            )
        else:
            return Exception(
                "🚫 Rate limit exceeded: Too many requests to Gemini API. "
                "Please wait a moment before trying again."
            )
    
    # Handle authentication errors
    elif "400" in error_str or "unauthorized" in error_str.lower():
        return Exception(
            "🔑 Authentication failed: Invalid or missing Google API key. "
            "Please check your GOOGLE_API_KEY environment variable."
        )
    
    # Handle model not found errors
    elif "404" in error_str or "not found" in error_str.lower():
        return Exception(
            "🤖 Model not found: The specified Gemini model is not available. "
            "Please check the model name in your configuration."
        )
    
    # Handle quota/billing errors
    elif "quota" in error_str.lower() or "billing" in error_str.lower():
        return Exception(
            "💰 Billing/Quota issue: There's a problem with your Gemini API billing or quota. "
            "Please check your Google Cloud billing account."
        )
    
    # Handle permission errors
    elif "403" in error_str or "permission" in error_str.lower():
        return Exception(
            "🚫 Permission denied: You don't have permission to access this Gemini model. "
            "Please check your API key permissions."
        )
    
    # Generic error handling
    else:
        return Exception(
            f"🤖 Gemini API error: {error_str[:200]}{'...' if len(error_str) > 200 else ''}"
        )
