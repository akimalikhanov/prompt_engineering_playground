import importlib
from typing import Any, Dict, List

import requests

from utils.load_configs import _load_tools_registry


def get_custom_movie_details(title: str, api_key: str = None, plot: str = "full"):
    import os

    if api_key is None:
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("OMDB_API_KEY", "")
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}&plot={plot}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "True":
            custom_json = {
                "Title": data.get("Title"),
                "Year": data.get("Year"),
                "Rated": data.get("Rated"),
                "Release Date": data.get("Released"),
                "Runtime": data.get("Runtime"),
                "Genre": data.get("Genre"),
                "Director": data.get("Director"),
                "Actors": data.get("Actors"),
                "Plot": data.get("Plot"),
                "Country": data.get("Country"),
                "Awards": data.get("Awards"),
                "Ratings": {
                    r["Source"]: r["Value"] for r in data.get("Ratings", [])
                },
                "imdb_link": f"https://www.imdb.com/title/{data.get('imdbID')}/"
                if data.get("imdbID")
                else None,
                "Box office (USA)": data.get("BoxOffice"),
            }
            return custom_json
        else:
            return {"error": data.get("Error", "Unknown error")}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}


# -----------------------------------------------------------------------------
# Tool registry helpers
# -----------------------------------------------------------------------------

def get_tool_schema(name: str) -> Dict[str, Any]:
    """Return the OpenAI-style tool schema (without python metadata) for a tool."""
    registry = _load_tools_registry()
    tool = registry.get(name)
    if not tool:
        raise KeyError(f"Tool '{name}' not found in tools registry")

    # Strip out python-specific metadata before giving to the LLM API.
    return {k: v for k, v in tool.items() if k != "python"}


def get_all_tool_schemas() -> List[Dict[str, Any]]:
    """Return all tool schemas defined in the YAML registry (without python metadata)."""
    registry = _load_tools_registry()
    return [get_tool_schema(name) for name in registry.keys()]


def get_tool_callable(name: str):
    """Resolve the Python callable for a given tool name using the YAML registry."""
    registry = _load_tools_registry()
    tool = registry.get(name)
    if not tool:
        raise KeyError(f"Tool '{name}' not found in tools registry")

    python_meta = tool.get("python", {}) or {}
    module_name = python_meta.get("module", "services.llm_tools")
    func_name = python_meta.get("callable", name)

    module = importlib.import_module(module_name)
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Callable '{func_name}' not found in module '{module_name}' "
            f"for tool '{name}'"
        ) from exc
