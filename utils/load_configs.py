import yaml
from functools import lru_cache
from typing import Any, Dict, List
import os

# Get the project root (parent of utils directory)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_models_config():
    """Load and validate models configuration from YAML file."""
    try:
        config_path = os.path.join(_PROJECT_ROOT, 'config', 'models.yaml')
        with open(config_path, 'r') as f:
            models_config = yaml.safe_load(f)
        return models_config
    except FileNotFoundError:
        raise FileNotFoundError("Error: models.yaml not found in config/ directory")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error: Invalid YAML format in models.yaml: {e}")


def _load_tools_config() -> Dict[str, Any]:
    """Load tools registry configuration from YAML file."""
    try:
        config_path = os.path.join(_PROJECT_ROOT, "config", "tools.yaml")
        with open(config_path, "r") as f:
            tools_config = yaml.safe_load(f) or {}
        return tools_config
    except FileNotFoundError:
        raise FileNotFoundError("Error: tools.yaml not found in config/ directory")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error: Invalid YAML format in tools.yaml: {e}")


def _load_gradio_templates():
    """Load and validate gradio templates configuration from YAML file."""
    try:
        config_path = os.path.join(_PROJECT_ROOT, 'config', 'gradio_templates.yaml')
        with open(config_path, 'r') as f:
            templates_config = yaml.safe_load(f)
        return templates_config
    except FileNotFoundError:
        raise FileNotFoundError("Error: gradio_templates.yaml not found in config/ directory")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error: Invalid YAML format in gradio_templates.yaml: {e}")


@lru_cache(maxsize=1)
def _load_tools_registry() -> Dict[str, Any]:
    """Load tools registry as a lookup dict keyed by tool/function name.

    The underlying YAML is expected to have the shape:
        tools:
          - type: function
            function:
              name: ...
              description: ...
              parameters: ...
            python:
              module: ...
              callable: ...
    """
    config = _load_tools_config()
    tools_list = config.get("tools", []) or []

    registry: Dict[str, Any] = {}
    for tool in tools_list:
        fn = (tool.get("function") or {}).get("name")
        if not fn:
            # Skip invalid entries without a function name
            continue
        registry[fn] = tool
    return registry


@lru_cache(maxsize=1)
def _load_models() -> Dict[str, Any]:
    """Load and transform models configuration for UI use."""
    config = _load_models_config()
    models: List[Dict[str, Any]] = []
    for model_cfg in config.get("models", []):
        models.append(
            {
                "id": model_cfg["id"],
                "label": model_cfg["label"],
                "provider": model_cfg["provider"],
                "params_override": model_cfg.get("params_override", {}),
            }
        )
    defaults = config.get("defaults", {}).get("params", {})
    return {
        "models": models,
        "defaults": defaults,
    }


@lru_cache(maxsize=1)
def _load_endpoints() -> Dict[str, Dict[str, Any]]:
    """Load endpoint configurations from gradio_templates.yaml."""
    templates_config = _load_gradio_templates()
    endpoints_raw = templates_config.get("endpoints", {})
    
    # Process endpoints - convert boolean strings if needed and ensure proper types
    endpoints: Dict[str, Dict[str, Any]] = {}
    for key, value in endpoints_raw.items():
        endpoints[key] = {
            "label": value.get("label", ""),
            "path": value.get("path", ""),
            "stream": bool(value.get("stream", False)),
            "sse": bool(value.get("sse", False)),
        }
    
    return endpoints


@lru_cache(maxsize=1)
def _get_default_endpoint_key() -> str:
    """Get the default endpoint key from gradio_templates.yaml."""
    templates_config = _load_gradio_templates()
    default_endpoint = templates_config.get("default_endpoint")
    
    if default_endpoint:
        endpoints = _load_endpoints()
        # Verify the default endpoint exists
        if default_endpoint in endpoints:
            return default_endpoint
    
    # Fallback: return first endpoint key if default not found
    endpoints = _load_endpoints()
    if endpoints:
        return next(iter(endpoints.keys()))
    
    raise RuntimeError("No endpoints configured in config/gradio_templates.yaml")


@lru_cache(maxsize=1)
def _load_delimiter_definitions() -> Dict[str, Dict[str, Any]]:
    """Load delimiter definitions from YAML and map resolver types to functions."""
    # Import here to avoid circular dependency
    from utils.delimiter_utils import RESOLVER_FUNCTIONS
    
    templates_config = _load_gradio_templates()
    delimiters_raw = templates_config.get("delimiters", {})
    
    # Process each delimiter definition
    delimiter_definitions: Dict[str, Dict[str, Any]] = {}
    for key, value in delimiters_raw.items():
        definition = dict(value)  # Make a copy
        
        # If there's a resolver_type, map it to the actual resolver function
        resolver_type = definition.pop("resolver_type", None)
        if resolver_type:
            resolver_func = RESOLVER_FUNCTIONS.get(resolver_type)
            if resolver_func:
                definition["resolver"] = resolver_func
            else:
                # Fallback: if resolver type not found, treat as static
                print(f"Warning: Unknown resolver_type '{resolver_type}' for delimiter '{key}'")
        
        delimiter_definitions[key] = definition
    
    return delimiter_definitions


@lru_cache(maxsize=1)
def _load_response_schema_templates() -> Dict[str, Dict[str, Any]]:
    """Load response schema templates from YAML config."""
    templates_config = _load_gradio_templates()
    templates_raw = templates_config.get("response_schema_templates", [])
    templates_lookup: Dict[str, Dict[str, Any]] = {}
    
    for template in templates_raw:
        label = template.get("label", template.get("id", "Unknown"))
        templates_lookup[label] = template
    
    # Create a lookup dictionary by label, with "Custom..." option
    templates_lookup.update({
        "Custom...": {
            "id": "custom",
            "label": "Custom...",
            "custom": True,
        }
    })
    return templates_lookup

