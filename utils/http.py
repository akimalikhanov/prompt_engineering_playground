from typing import Optional

def resolve_api_base(api_base_url: Optional[str], default_base: str) -> str:
    api_url = (api_base_url or default_base).strip()
    return api_url or default_base


