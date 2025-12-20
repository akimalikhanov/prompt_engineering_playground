"""
Utility for rendering Jinja2 templates in prompt messages
"""

import re
from typing import Any

from jinja2 import Environment, StrictUndefined, TemplateSyntaxError, UndefinedError, meta


def extract_variables_from_template(template_str: str) -> list[str]:
    """
    Extract all variable names from a Jinja2 template string.

    Args:
        template_str: Template string

    Returns:
        List of variable names
    """
    try:
        env = Environment()
        ast = env.parse(template_str)
        return sorted(meta.find_undeclared_variables(ast))
    except Exception:
        # Fallback: simple regex for {{variable}} patterns
        matches = re.findall(r"\{\{\s*(\w+)(?:\s*\|[^}]*)?\s*\}\}", template_str)
        return sorted(set(matches))


def render_template(
    template_str: str, variables: dict[str, Any]
) -> tuple[str, list[str], list[str]]:
    """
    Render a Jinja2 template with provided variables.

    Args:
        template_str: Template string to render
        variables: Dict of variables to use in rendering

    Returns:
        Tuple of (rendered_string, missing_vars, warnings)
    """
    warnings = []
    missing_vars = []

    try:
        # Extract expected variables
        expected_vars = extract_variables_from_template(template_str)

        # Check for missing required variables
        provided_vars = set(variables.keys())
        missing_vars = [v for v in expected_vars if v not in provided_vars]

        # Create Jinja environment with strict undefined
        env = Environment(undefined=StrictUndefined)
        template = env.from_string(template_str)

        # Render
        rendered = template.render(**variables)

        # Check for unused variables
        unused_vars = provided_vars - set(expected_vars)
        if unused_vars:
            warnings.append(f"Unused variables: {', '.join(sorted(unused_vars))}")

        return rendered, missing_vars, warnings

    except UndefinedError as e:
        # This shouldn't happen if we checked missing_vars correctly, but just in case
        return template_str, missing_vars, [f"Template error: {str(e)}"]

    except TemplateSyntaxError as e:
        return template_str, missing_vars, [f"Template syntax error: {str(e)}"]

    except Exception as e:
        return template_str, missing_vars, [f"Rendering error: {str(e)}"]


def render_messages(
    messages: list[dict[str, str]], variables: dict[str, Any]
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    """
    Render all message contents with Jinja2.

    Args:
        messages: List of message dicts with 'role' and 'content'
        variables: Dict of variables to use in rendering

    Returns:
        Tuple of (rendered_messages, missing_vars_all, warnings_all)
    """
    rendered_messages = []
    all_missing_vars = set()
    all_warnings = []
    all_used_vars = set()  # Track variables used across ALL messages
    provided_vars = set(variables.keys())

    # First pass: collect all variables used across all messages
    for msg in messages:
        content = msg.get("content", "")
        expected_vars = extract_variables_from_template(content)
        all_used_vars.update(expected_vars)

    # Second pass: render messages and check for missing variables per message
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        rendered_content, missing_vars, warnings = render_template(content, variables)

        # Collect missing vars (still useful per-message)
        all_missing_vars.update(missing_vars)

        # Only add template/syntax errors to warnings (not unused vars per message)
        for w in warnings:
            if not w.startswith("Unused variables"):
                all_warnings.append(f"Message {i} ({msg.get('role', 'unknown')}): {w}")

        # Build rendered message
        rendered_msg = {"role": msg.get("role", "user"), "content": rendered_content}
        rendered_messages.append(rendered_msg)

    # Check for variables unused across ALL messages
    unused_vars = provided_vars - all_used_vars
    if unused_vars:
        all_warnings.append(f"Unused variables: {', '.join(sorted(unused_vars))}")

    return rendered_messages, sorted(all_missing_vars), all_warnings
