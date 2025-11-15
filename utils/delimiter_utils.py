from typing import Any, Dict, Tuple


def _resolve_custom_delimiter(cs: str, ce: str) -> tuple[str, str]:
    start_token = (cs or "<<<CUSTOM>>>").rstrip()
    end_token = (ce or "<<<END>>>").lstrip()
    start = f"{start_token}\n"
    end = f"\n{end_token}"
    return start, end


def _resolve_markdown_delimiter(cs: str, _: str) -> tuple[str, str]:
    language = cs.strip()
    prefix = f"```{language}\n" if language else "```\n"
    return prefix, "\n```"


def _resolve_xml_delimiter(cs: str, _: str) -> tuple[str, str]:
    tag = cs.strip() or "section"
    return f"<{tag}>\n", f"\n</{tag}>"


# Resolver functions mapped by type
RESOLVER_FUNCTIONS: Dict[str, Any] = {
    "markdown": _resolve_markdown_delimiter,
    "xml": _resolve_xml_delimiter,
    "custom": _resolve_custom_delimiter,
}


def _delim_pair(style: str, custom_start: str, custom_end: str, delimiter_definitions: Dict[str, Dict[str, Any]]) -> tuple[str, str]:
    """Get delimiter pair for a given style.
    
    Args:
        style: Delimiter style name
        custom_start: Custom start token
        custom_end: Custom end token
        delimiter_definitions: Dict of delimiter definitions
        
    Returns:
        Tuple of (start_delimiter, end_delimiter)
    """
    style = (style or "").strip()
    cs = (custom_start or "").strip()
    ce = (custom_end or "").strip()

    definition = delimiter_definitions.get(style)
    if not definition:
        return "```\n", "\n```"

    resolver = definition.get("resolver")
    if resolver:
        return resolver(cs, ce)

    return definition["start"], definition["end"]


def _insert_pair_at_end(current_text: str, style: str, cs: str, ce: str, delimiter_definitions: Dict[str, Dict[str, Any]]) -> str:
    start, end = _delim_pair(style, cs, ce, delimiter_definitions)
    current_text = current_text or ""
    # add a blank line before for readability unless already at a newline or empty
    prefix = "" if (not current_text or current_text.endswith("\n")) else "\n"
    return f"{current_text}{prefix}{start}{end}"


def _wrap_entire_message(current_text: str, style: str, cs: str, ce: str, delimiter_definitions: Dict[str, Dict[str, Any]]) -> str:
    start, end = _delim_pair(style, cs, ce, delimiter_definitions)
    current_text = current_text or ""
    return f"{start}{current_text}\n{end}"

