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
    
    # Special handling for models that only support default temperature=1.0
    # Note: `model_id` is the registry id from config/models.yaml (not the provider `model_name`).
    if model_id in {"gpt-o4-mini", "gpt-5-mini", "gpt-5-nano", "gpt-5.2"}:
        special_behaviors["temperature"] = {
            "value": 1.0,
            "interactive": False,
            "label": "Temperature (fixed at 1 for this model)"
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

    # Reasoning effort and verbosity (only visible for GPT-5 family)
    gpt5_models = {"gpt-5.2", "gpt-5-mini", "gpt-5-nano"}
    is_gpt5 = model_id in gpt5_models if model_id else False

    # Reasoning choices depend on model support
    if model_id == "gpt-5.2":
        reasoning_choices = ["none", "low", "medium", "high"]
    elif model_id in {"gpt-5-mini", "gpt-5-nano"}:
        reasoning_choices = ["minimal", "low", "medium", "high"]
    else:
        reasoning_choices = ["none", "minimal", "low", "medium", "high"]

    reasoning_effort_value = effective.get("reasoning_effort", defaults.get("reasoning_effort"))
    # Only set default for GPT-5 models; keep None for others to avoid sending unsupported params
    if reasoning_effort_value is None and is_gpt5:
        reasoning_effort_value = "low"
    if reasoning_effort_value is not None and reasoning_effort_value not in reasoning_choices:
        reasoning_effort_value = reasoning_choices[0]
    params["reasoning_effort"] = {
        "value": reasoning_effort_value,
        "interactive": True,
        "visible": is_gpt5,
        "choices": reasoning_choices,
    }

    verbosity_value = effective.get("verbosity", defaults.get("verbosity"))
    # Only set default for GPT-5 models; keep None for others to avoid sending unsupported params
    if verbosity_value is None and is_gpt5:
        verbosity_value = "low"
    params["verbosity"] = {
        "value": verbosity_value,
        "interactive": True,
        "visible": is_gpt5,
    }

    return params

