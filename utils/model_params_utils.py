from typing import Any, Dict, Optional


def get_effective_params_for_model(model_id: Optional[str], models_data: Dict[str, Any], model_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Get effective parameters for a model by merging defaults with params_override.
    
    Args:
        model_id: The model ID, or None to get just defaults
        models_data: Models data dict with "defaults" key
        model_lookup: Dict mapping model_id to model config
        
    Returns:
        Dict with effective parameter values (temperature, top_p, max_tokens, seed, etc.)
    """
    # Start with defaults from models.yaml
    effective = models_data.get("defaults", {}).copy()
    
    # Apply model-specific overrides if any
    if model_id:
        model_config = model_lookup.get(model_id)
        if model_config:
            params_override = model_config.get("params_override", {})
            if params_override:
                effective.update(params_override)
    
    return effective


def _get_model_special_behavior(model_id: str) -> Dict[str, Any]:
    """Get special behavior configuration for a model (e.g., fixed parameters, disabled UI).
    
    This centralizes special-case handling for models that have constraints.
    Returns a dict mapping parameter names to their special behavior config.
    """
    special_behaviors: Dict[str, Dict[str, Any]] = {}
    
    # Special handling for o4-mini: temperature must be 1.0 and cannot be changed
    if model_id == "gpt-o4-mini":
        special_behaviors["temperature"] = {
            "value": 1.0,
            "interactive": False,
            "label": "Temperature (fixed at 1.0 for o4-mini)"
        }
    
    return special_behaviors


def _get_param_values_for_model(
    model_id: Optional[str],
    models_data: Dict[str, Any],
    model_lookup: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Get parameter values for a model, including special behaviors.
    
    Returns a dict with parameter values and UI properties (like interactive, label).
    Used by both initialization and update functions.
    
    Args:
        model_id: The model ID
        models_data: Models data dict with "defaults" key
        model_lookup: Dict mapping model_id to model config
        
    Returns:
        Dict with parameter configurations for UI components
    """
    defaults = models_data.get("defaults", {})
    effective = get_effective_params_for_model(model_id, models_data, model_lookup)
    special_behaviors = _get_model_special_behavior(model_id) if model_id else {}
    
    params = {}
    
    # Temperature
    temp_value = effective.get("temperature", defaults.get("temperature"))
    temp_special = special_behaviors.get("temperature", {})
    if "value" in temp_special:
        temp_value = temp_special["value"]
    params["temperature"] = {
        "value": temp_value,
        "interactive": temp_special.get("interactive", True),
        "label": temp_special.get("label", "Temperature")
    }
    
    # Top-p
    params["top_p"] = {
        "value": effective.get("top_p", defaults.get("top_p"))
    }
    
    # Max tokens
    max_tokens_value = effective.get("max_tokens", defaults.get("max_tokens"))
    if max_tokens_value is not None:
        max_tokens_value = int(max_tokens_value)
    params["max_tokens"] = {"value": max_tokens_value}
    
    # Seed
    seed_value = effective.get("seed", defaults.get("seed"))
    params["seed"] = {"value": str(seed_value) if seed_value is not None else ""}
    
    return params

