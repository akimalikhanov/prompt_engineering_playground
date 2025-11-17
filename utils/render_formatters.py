import json
from typing import Dict, List, Tuple

def format_prompt_details(prompt: Dict) -> Tuple[str, str]:
    variables = prompt.get("variables", [])
    default_vars = {}
    for var in variables:
        if isinstance(var, dict):
            name = var.get("name", "")
            default = var.get("default", "")
            if name:
                default_vars[name] = default

    desc_parts = []
    if prompt.get("description"):
        desc_parts.append(f"**Description:** {prompt.get('description')}")
    if prompt.get("category"):
        desc_parts.append(f"**Category:** {prompt.get('category')}")
    technique = prompt.get("technique") or prompt.get("technique_key")
    if technique:
        desc_parts.append(f"**Technique:** {technique}")

    description_text = "\n\n".join(desc_parts) if desc_parts else "No description available."
    return description_text, json.dumps(default_vars, indent=2)

def format_render_preview(rendered_messages: List[Dict], variables: Dict) -> str:
    variables_section = ""
    if isinstance(variables, dict) and variables:
        var_lines = [f"{k}: {variables[k]}" for k in sorted(variables.keys())]
        variables_section = "Variables:\n" + "\n".join(var_lines) + "\n\n"

    blocks = []
    for msg in rendered_messages:
        role = (msg.get("role") or "unknown").upper()
        content = msg.get("content", "")
        blocks.append(f"[{role}]\n{content}")

    return variables_section + "\n\n".join(blocks)

def format_render_status(missing_vars: List[str], warnings: List[str]) -> str:
    parts: List[str] = []
    if missing_vars:
        parts.append(f"⚠️ Missing variables: {', '.join(missing_vars)}")
    if warnings:
        parts.extend([f"⚠️ {w}" for w in warnings])
    if not parts:
        parts.append("✅ Template rendered successfully!")
    return "\n".join(parts)


