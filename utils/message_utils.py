import json
from typing import Any, Dict, List, Optional, Tuple


def _build_messages(
    user_message: str,
    history: List[Tuple[str, str]],
    system_prompt: Optional[str],
    context_prompt: Optional[str],
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    if context_prompt and context_prompt.strip():
        messages.append({"role": "user", "content": f"CONTEXT:\n{context_prompt.strip()}\n"})
    for past_user, past_assistant in history:
        if past_user:
            messages.append({"role": "user", "content": past_user})
        if past_assistant:
            messages.append({"role": "assistant", "content": past_assistant})
    messages.append({"role": "user", "content": user_message})
    return messages


def _coerce_seed(seed_text: Optional[str]) -> Optional[int]:
    if seed_text is None:
        return None
    if isinstance(seed_text, (int, float)):
        try:
            seed_val = int(seed_text)
        except ValueError:
            return None
        return max(0, seed_val)
    seed_text = seed_text.strip()
    if not seed_text:
        return None
    try:
        seed_val = int(seed_text)
    except ValueError:
        return None
    return max(0, seed_val)


def _build_response_format(
    mode: str,
    strict: bool,
    schema_code: Optional[str],
    schema_template_label: Optional[str] = None,
    response_schema_templates: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Build response_format dict from UI inputs.
    
    Expected structure for json_schema type:
    {
        "type": "json_schema",
        "json_schema": {
            "schema": {...},  # The actual JSON schema
            "strict": true,
            "name": "SchemaName"  # Required by OpenAI
        }
    }
    
    Args:
        mode: Response format mode ("None", "JSON object", "JSON schema")
        strict: Whether to use strict schema validation
        schema_code: JSON schema code string
        schema_template_label: Selected schema template label
        response_schema_templates: Dict of response schema templates (optional)
    """
    if mode == "None" or not mode:
        return None
    
    if mode == "JSON object":
        return {"type": "json_object"}
    
    if mode == "JSON schema":
        if not schema_code or not schema_code.strip():
            return None
        try:
            # Parse and validate the JSON schema
            schema_text = schema_code.strip()
            json_schema_dict = json.loads(schema_text)
            
            # Validate it's actually a dict/object
            if not isinstance(json_schema_dict, dict):
                print(f"⚠️ JSON schema must be an object, got: {type(json_schema_dict)}")
                return None
            
            # Extract schema name from template if available
            schema_name = None
            if response_schema_templates and schema_template_label and schema_template_label != "Custom...":
                template = response_schema_templates.get(schema_template_label)
                if template:
                    json_schema_config = template.get("json_schema", {})
                    schema_name = json_schema_config.get("name")
            
            # Build json_schema structure - name is required by OpenAI
            # Use template name if available, otherwise default to "CustomSchema"
            json_schema_obj = {
                "schema": json_schema_dict,
                "strict": strict,
                "name": schema_name if schema_name else "CustomSchema",  # Required by OpenAI API
            }
            
            # Wrap in the expected structure
            return {
                "type": "json_schema",
                "json_schema": json_schema_obj,
            }
        except json.JSONDecodeError as e:
            # Log the error so user can see it in console/logs
            error_msg = f"⚠️ Invalid JSON schema: {str(e)}"
            print(error_msg)
            # Still return None to skip invalid schema
            return None
    
    return None


def _normalize_selection(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("value") or value.get("id")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[1]
    return None

