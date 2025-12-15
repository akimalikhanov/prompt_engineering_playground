import os
from dotenv import load_dotenv
from adapters.openai_adapter import _unified_openai_api_call
from utils.errors import (_get_status_code, \
                    prettify_openai_error, \
                    BackendError)
from utils.retry import call_with_retry
from utils.load_configs import _load_models_config, expand_env_var
import time
import logging


logger = logging.getLogger(os.getenv("LOGGER_NAME", "llm-router"))
# Load environment variables
load_dotenv()

# Load configuration once at module level
_models_config = _load_models_config()
_models_lookup = {model['id']: model for model in _models_config['models']}


def route_call(
    provider_id,
    model_id,
    messages,
    params,
    stream=True,
    stream_mode="sse",
    max_retries=3,
    trace_id=None,
    session_id=None,
    backend_endpoint=None,
):
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
    
    # Merge parameters: defaults -> model overrides -> request params (request wins)
    # IMPORTANT: request params that are null/None should NOT override defaults.
    merged_params = _models_config["defaults"]["params"].copy()
    if "params_override" in model_config:
        merged_params.update(model_config["params_override"])

    request_params = params or {}
    if isinstance(request_params, dict):
        request_params = {k: v for k, v in request_params.items() if v is not None}
    merged_params.update(request_params)

    # Tools are currently supported only for non-streaming calls.
    # tools = merged_params.get("tools")
    # if tools and stream:
    #    raise ValueError("Tools/functions are only supported for non-streaming calls. "
    #                     "Use the /chat endpoint (non-stream) when tools are enabled.")
    
    # Resolve base_url and api_key per provider and forward through openai_adapter
    providers = _models_config['providers']
    if provider_id not in providers:
        available_providers = list(providers.keys())
        raise ValueError(f"Provider ID '{provider_id}' not supported. Available providers: {available_providers}")

    provider_cfg = providers[provider_id]

    # Determine base_url and model
    if provider_id == 'vllm':
        server_config = model_config['server']
        # Expand environment variables in server config
        host = expand_env_var(server_config['host'])
        port = expand_env_var(server_config['port'])
        base_url = f"http://{host}:{port}{server_config['base_path']}"
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
        return call_with_retry(
            lambda: _unified_openai_api_call(
                model=model_name,
                messages=normalized_messages,
                params=merged_params,
                stream=stream,
                api_key=api_key,
                base_url=base_url,
                stream_mode=stream_mode,
                provider_id=provider_id,
                model_config_id=model_id,
                session_id=session_id,
                trace_id=trace_id,
                backend_endpoint=backend_endpoint,
            ),
            max_retries=max_retries,
            base_delay=0.5,
            max_delay=8.0,
            jitter_max=0.5,
            logger=logger,
            # corr_id is auto-generated; optionally pass one from your request context
        )
    except Exception as e:
        msg = prettify_openai_error(e)
        status = _get_status_code(e)
        # Preserve your rule: 4xx → pass through; else 502
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
