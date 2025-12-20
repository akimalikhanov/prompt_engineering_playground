import json


def build_schema_from_template(
    template_label: str, templates: dict, current_strict: bool
) -> tuple[str, bool]:
    if not template_label or template_label == "Custom...":
        default_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": [],
        }
        return json.dumps(default_schema, indent=2), current_strict

    template = templates.get(template_label)
    if not template:
        default_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": [],
        }
        return json.dumps(default_schema, indent=2), current_strict

    json_schema_config = template.get("json_schema", {})
    schema_dict = json_schema_config.get("schema", {})
    return json.dumps(schema_dict, indent=2), current_strict


def extract_response_format_from_prompt(prompt: dict) -> tuple[str, str]:
    """Map backend response_format fields into UI response_mode and schema_code."""
    response_mode = "None"
    schema_code = ""
    response_format = prompt.get("response_format")
    if response_format == "json_object":
        response_mode = "JSON object"
    elif response_format == "json_schema":
        response_mode = "JSON schema"
        json_schema_template = prompt.get("json_schema_template")
        if json_schema_template:
            schema_code = json.dumps(json_schema_template, indent=2)
    return response_mode, schema_code
