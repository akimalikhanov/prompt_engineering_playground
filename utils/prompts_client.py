import httpx
from typing import Dict, Any, List

def list_prompts(api_base_url: str) -> List[Dict[str, Any]]:
    r = httpx.get(f"{api_base_url.rstrip('/')}/prompts", timeout=10.0)
    r.raise_for_status()
    return r.json().get("prompts", [])

def get_prompt(api_base_url: str, prompt_id: str) -> Dict[str, Any]:
    r = httpx.get(f"{api_base_url.rstrip('/')}/prompts/{prompt_id}", timeout=10.0)
    r.raise_for_status()
    return r.json()

def render_prompt(api_base_url: str, prompt_id: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    r = httpx.post(
        f"{api_base_url.rstrip('/')}/prompts/{prompt_id}/render",
        json={"variables": variables},
        timeout=10.0,
    )
    r.raise_for_status()
    return r.json()


