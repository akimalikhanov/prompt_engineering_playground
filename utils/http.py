def resolve_api_base(api_base_url: str | None, default_base: str) -> str:
    api_url = (api_base_url or default_base).strip()
    return api_url or default_base
