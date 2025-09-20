import os
import yaml
from dotenv import load_dotenv
from types import GeneratorType
from adapters.openai_conn import openai_call
from adapters.gemini_conn import gemini_call
from adapters.vllm_conn import vllm_call

# Load environment variables
load_dotenv()

# Load model configurations from YAML file
def _load_models_config():
    """Load and validate models configuration from YAML file."""
    try:
        with open('config/models.yaml', 'r') as f:
            models_config = yaml.safe_load(f)
        return models_config
    except FileNotFoundError:
        raise FileNotFoundError("Error: models.yaml not found in config/ directory")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error: Invalid YAML format in models.yaml: {e}")

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
    
    # Route to appropriate adapter based on provider_id
    if provider_id == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        return openai_call(
            model=model_config['model_name'],
            messages=messages,
            params=merged_params,
            stream=stream,
            api_key=api_key
        )
    
    elif provider_id == 'google':
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        return gemini_call(
            model=model_config['model_name'],
            messages=messages,
            params=merged_params,
            stream=stream,
            api_key=api_key
        )
    
    elif provider_id == 'vllm':
        # Build base URL for vLLM server
        server_config = model_config['server']
        base_url = f"http://{server_config['host']}:{server_config['port']}{server_config['base_path']}"
        
        return vllm_call(
            model=model_config['model_path'],
            messages=messages,
            params=merged_params,
            stream=stream,
            base_url=base_url
        )
    
    else:
        available_providers = list(_models_config['adapters'].keys())
        raise ValueError(f"Provider ID '{provider_id}' not supported. Available providers: {available_providers}")


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
    return _models_config['adapters'].copy()


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
