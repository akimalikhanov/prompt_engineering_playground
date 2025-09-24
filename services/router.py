import os
from dotenv import load_dotenv
from types import GeneratorType
from adapters.openai_adapter import _unified_openai_api_call, prettify_openai_error
from services.errors import BackendError
from config.load_models_config import _load_models_config

# Load environment variables
load_dotenv()

# Load configuration once at module level
_models_config = _load_models_config()
_models_lookup = {model['id']: model for model in _models_config['models']}

def route_call(provider_id, model_id, messages, params, stream=True):
    """
    Route API calls to the appropriate adapter based on provider and model.
    
    Args:
        provider_id (str): Provider identifier (e.g., 'openai', 'google', 'vllm')
        model_id (str): Model identifier (e.g., 'gpt', 'gemini', 'llama', 'qwen')
        messages: Message content or list of messages
        params (dict): Dictionary containing parameters like temperature, top_p, max_tokens, seed
        stream (bool): If True, returns streaming response. If False, returns complete response.
    
    Returns:
        If stream=True: Generator yielding streaming response chunks
        If stream=False: String containing the complete response
    
    Raises:
        ValueError: If provider_id or model_id is not found in configuration
        KeyError: If required configuration is missing
    """
    # Validate model_id exists in configuration
    if model_id not in _models_lookup:
        available_models = list(_models_lookup.keys())
        raise ValueError(f"Model ID '{model_id}' not found. Available models: {available_models}")
    
    model_config = _models_lookup[model_id]
    
    # Validate provider_id matches the model's provider
    if model_config['provider'] != provider_id:
        raise ValueError(f"Provider ID '{provider_id}' does not match model '{model_id}' provider '{model_config['provider']}'")
    
    # Merge default parameters with provided parameters
    merged_params = _models_config['defaults']['params'].copy()
    merged_params.update(params)
    
    # Apply model-specific parameter overrides if any
    if 'params_override' in model_config:
        merged_params.update(model_config['params_override'])
    
    # Resolve base_url and api_key per provider and forward through openai_adapter
    providers = _models_config['providers']
    if provider_id not in providers:
        available_providers = list(providers.keys())
        raise ValueError(f"Provider ID '{provider_id}' not supported. Available providers: {available_providers}")

    provider_cfg = providers[provider_id]

    # Determine base_url and model
    if provider_id == 'vllm':
        server_config = model_config['server']
        base_url = f"http://{server_config['host']}:{server_config['port']}{server_config['base_path']}"
        model_name = model_config['model_path']
        api_key = "dummy"
    else:
        base_url = provider_cfg['base_url']
        model_name = model_config['model_name']
        api_key_env = provider_cfg.get('api_key_env')
        api_key = os.getenv(api_key_env) if api_key_env else None
        if not api_key:
            raise ValueError(f"{api_key_env} environment variable not set")

    # Normalize messages: accept str or pre-built list
    normalized_messages = (
        [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
    )
    try:
        return _unified_openai_api_call(
            model=model_name,
            messages=normalized_messages,
            params=merged_params,
            stream=stream,
            api_key=api_key,
            base_url=base_url,
        )
    except Exception as e:
        msg = prettify_openai_error(e)
        status = getattr(e, "status_code", None)
        status_code = status if isinstance(status, int) and 400 <= status < 500 else 502
        raise BackendError(msg, status_code)


def get_available_models():
    """
    Get list of available model configurations.
    
    Returns:
        dict: Dictionary mapping model_id to model configuration
    """
    return _models_lookup.copy()


def get_available_providers():
    """
    Get list of available provider configurations.
    
    Returns:
        dict: Dictionary mapping provider_id to provider configuration
    """
    return _models_config['providers'].copy()


def get_default_params():
    """
    Get default parameters from configuration.
    
    Returns:
        dict: Dictionary containing default parameters
    """
    return _models_config['defaults']['params'].copy()


def get_default_stream_setting():
    """
    Get default streaming setting from configuration.
    
    Returns:
        bool: Default streaming setting
    """
    return _models_config['defaults']['stream']
