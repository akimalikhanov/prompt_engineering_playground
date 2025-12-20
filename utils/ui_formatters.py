from typing import Any


def _render_badge(label: str, value: str | None) -> str:
    display = value if value is not None else "—"
    return f"<span class='badge'><span class='badge-label'>{label}</span><span class='badge-value'>{display}</span></span>"


def _format_tokens(metrics: dict[str, Any]) -> str | None:
    """Format token counts as: total (in prompt, out completion, reason reasoning)."""
    prompt = metrics.get("prompt_tokens")
    completion = metrics.get("completion_tokens")
    reasoning = metrics.get("reasoning_tokens")
    total = metrics.get("total_tokens")
    if prompt is None and completion is None and total is None:
        return None

    # Build the breakdown parts
    breakdown_parts = []
    if prompt is not None:
        breakdown_parts.append(f"in {prompt}")
    if completion is not None:
        breakdown_parts.append(f"out {completion}")
    if reasoning is not None:
        breakdown_parts.append(f"r {reasoning}")

    # Format: total (in X, out Y, reason Z) or just total if no breakdown available
    if breakdown_parts:
        breakdown_str = " | ".join(breakdown_parts)
        if total is not None:
            return f"{total} ({breakdown_str})"
        return f"({breakdown_str})"

    # Fallback: just show total if available
    return str(total) if total is not None else None


def _format_cost(metrics: dict[str, Any]) -> str | None:
    cost = metrics.get("cost_usd")
    if cost is None:
        return None
    return f"${cost:.4f}"


def _format_tokens_per_second(metrics: dict[str, Any]) -> str | None:
    tokens_per_second = metrics.get("tokens_per_second")
    if tokens_per_second is None:
        return None
    # Round to 1 decimal place, but show as integer if it's a whole number
    rounded = round(tokens_per_second, 1)
    if rounded == int(rounded):
        return f"{int(rounded)} tps"
    return f"{rounded:.1f} tps"


def format_metrics_badges(
    metrics: dict[str, Any] | None, ttft_fallback: float | None, latency_fallback: float | None
) -> str:
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
