from typing import Any, Dict, Optional


def _render_badge(label: str, value: Optional[str]) -> str:
    display = value if value is not None else "—"
    return f"<span class='badge'><span class='badge-label'>{label}</span><span class='badge-value'>{display}</span></span>"


def _format_tokens(metrics: Dict[str, Any]) -> Optional[str]:
    prompt = metrics.get("prompt_tokens")
    completion = metrics.get("completion_tokens")
    total = metrics.get("total_tokens")
    if prompt is None and completion is None and total is None:
        return None
    parts = []
    if prompt is not None or completion is not None:
        prompt_val = prompt if prompt is not None else "?"
        completion_val = completion if completion is not None else "?"
        parts.append(f"{prompt_val}/{completion_val}")
    if total is not None:
        parts.append(f"({total})")
    return " ".join(parts) if parts else str(total)


def _format_cost(metrics: Dict[str, Any]) -> Optional[str]:
    cost = metrics.get("cost_usd")
    if cost is None:
        return None
    return f"${cost:.4f}"


def _format_tokens_per_second(metrics: Dict[str, Any]) -> Optional[str]:
    tokens_per_second = metrics.get("tokens_per_second")
    if tokens_per_second is None:
        return None
    # Round to 1 decimal place, but show as integer if it's a whole number
    rounded = round(tokens_per_second, 1)
    if rounded == int(rounded):
        return f"{int(rounded)} tps"
    return f"{rounded:.1f} tps"


def format_metrics_badges(metrics: Optional[Dict[str, Any]], ttft_fallback: Optional[float], latency_fallback: Optional[float]) -> str:
    metrics = metrics or {}
    ttft_ms = metrics.get("ttft_ms") or ttft_fallback
    latency_ms = metrics.get("latency_ms") or latency_fallback

    badges = [
        _render_badge("TTFT", f"{ttft_ms:.0f} ms" if ttft_ms is not None else None),
        _render_badge("Latency", f"{latency_ms:.0f} ms" if latency_ms is not None else None),
        _render_badge("Speed", _format_tokens_per_second(metrics)),
        _render_badge("Tokens", _format_tokens(metrics)),
        _render_badge("Cost", _format_cost(metrics)),
    ]
    return f"<div class='badges'>{''.join(badges)}</div>"

