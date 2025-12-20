"""
OpenTelemetry Metrics instrumentation for the LLM playground.

Exports metrics via OTLP to the collector, which then exposes them to Prometheus.
"""

import os

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

from utils.otel_config import (
    get_otel_resource,
    get_otlp_metrics_endpoint,
    get_process_instance_id,
)

# ---------------------------------------------------------------------------
# Lazy initialization (singleton pattern)
# ---------------------------------------------------------------------------
_initialized: bool = False
_pid: int | None = None
_llm_requests_total: Counter | None = None
_llm_latency_seconds: Histogram | None = None
_llm_ttft_seconds: Histogram | None = None
_llm_tokens_per_second: Histogram | None = None
_llm_tokens_total: Counter | None = None
_llm_cost_usd_total: Counter | None = None
_llm_tool_calls_total: Counter | None = None
_runs_feedback_total: Counter | None = None
_http_requests_total: Counter | None = None
_meter_provider: MeterProvider | None = None
_metrics_resource: Resource | None = None


def _get_metrics_resource() -> Resource:
    """
    Build a metrics-only OTEL resource that includes a per-worker identifier.

    We intentionally avoid adding the worker id to the shared OTEL resource to
    prevent it from becoming a Loki log label. Metrics still need the worker id
    so Prometheus can differentiate data emitted by each Gunicorn worker.
    """
    global _metrics_resource

    if _metrics_resource is None:
        worker_id = get_process_instance_id()
        worker_resource = Resource.create(
            {
                # Standard OTEL attribute used by Prometheus exporter as a label
                "service.instance.id": worker_id,
            }
        )
        _metrics_resource = get_otel_resource().merge(worker_resource)

    return _metrics_resource


def init_metrics() -> None:
    """Initialize MeterProvider once, on first use, or re-initialize if PID changes.

    This handles Gunicorn's pre-fork model where each worker process needs its own
    MeterProvider with a live background thread for exporting metrics.
    """
    global \
        _initialized, \
        _llm_requests_total, \
        _llm_latency_seconds, \
        _llm_ttft_seconds, \
        _llm_tokens_per_second, \
        _llm_tokens_total, \
        _llm_cost_usd_total, \
        _llm_tool_calls_total, \
        _runs_feedback_total, \
        _http_requests_total, \
        _pid, \
        _meter_provider

    current_pid = os.getpid()

    # If initialized and in the same process, do nothing
    if _initialized and _pid == current_pid:
        return

    # We need to create a provider if:
    # 1. First initialization (_initialized is False)
    # 2. OR we are in a forked child (PID changed), so the inherited provider is invalid/broken
    need_new_provider = not _initialized or (_pid != current_pid)

    if need_new_provider:
        # Create our own provider - don't rely on global state which may be inherited from fork
        exporter = OTLPMetricExporter(endpoint=get_otlp_metrics_endpoint())
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=15000)
        _meter_provider = MeterProvider(
            resource=_get_metrics_resource(),
            metric_readers=[reader],
        )

        # Try to set it globally (may fail if already set, but that's OK)
        # try:
        #     metrics.set_meter_provider(_meter_provider)
        # except Exception:
        #     pass

    # Get meter from our own provider (not the global one which may be broken)
    if _meter_provider:
        meter = _meter_provider.get_meter("pep.llm", version="0.1.0")
    else:
        meter = metrics.get_meter("pep.llm", version="0.1.0")

    # Create metrics
    _llm_requests_total = meter.create_counter(
        name="pep_llm_requests_total",
        description="Total number of LLM API requests",
        unit="1",
    )

    _llm_latency_seconds = meter.create_histogram(
        name="pep_llm_latency_seconds",
        description="LLM request latency in seconds",
        unit="s",
    )

    _llm_ttft_seconds = meter.create_histogram(
        name="pep_llm_ttft_seconds",
        description="LLM time to first token in seconds",
        unit="s",
    )

    _llm_tokens_per_second = meter.create_histogram(
        name="pep_llm_tokens_per_second",
        description="LLM tokens generated per second",
        unit="1",
    )

    _llm_tokens_total = meter.create_counter(
        name="pep_llm_tokens_total",
        description="Total number of LLM tokens by type",
        unit="1",
    )

    _llm_cost_usd_total = meter.create_counter(
        name="pep_llm_cost_total",
        description="Total LLM cost in USD",
        unit="USD",
    )

    _llm_tool_calls_total = meter.create_counter(
        name="pep_llm_tool_calls_total",
        description="Total number of tool calls by model and tool name",
        unit="1",
    )

    _runs_feedback_total = meter.create_counter(
        name="pep_runs_feedback_total",
        description="Total number of user feedback submissions by model, provider, and feedback type",
        unit="1",
    )

    _http_requests_total = meter.create_counter(
        name="pep_http_requests_total",
        description="Total HTTP requests to the API",
        unit="1",
    )

    _initialized = True
    _pid = current_pid


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def record_llm_request(
    provider: str,
    model_id: str,
    status: str,
    endpoint: str,
) -> None:
    """
    Increment the pep_llm_requests_total counter.

    Args:
        provider: Provider identifier (e.g., 'openai', 'google', 'vllm')
        model_id: Model identifier (e.g., 'gpt-4o', 'gemini-flash')
        status: Request status ('ok' or 'error')
        endpoint: Backend endpoint (e.g., '/chat', '/chat.stream')
    """
    try:
        init_metrics()

        if _llm_requests_total:
            _llm_requests_total.add(
                1,
                attributes={
                    "provider": provider or "unknown",
                    "model_id": model_id or "unknown",
                    "status": status or "unknown",
                    "endpoint": endpoint or "unknown",
                },
            )
    except Exception:
        # Metrics recording should never crash the app
        pass


def record_llm_latency(
    provider: str,
    model_id: str,
    endpoint: str,
    latency_seconds: float,
) -> None:
    """
    Record LLM request latency in seconds.

    Args:
        provider: Provider identifier (e.g., 'openai', 'google', 'vllm')
        model_id: Model identifier (e.g., 'gpt-4o', 'gemini-flash')
        endpoint: Backend endpoint (e.g., '/chat', '/chat.stream')
        latency_seconds: Request latency in seconds
    """
    try:
        init_metrics()

        if _llm_latency_seconds and latency_seconds is not None:
            _llm_latency_seconds.record(
                latency_seconds,
                attributes={
                    "provider": provider or "unknown",
                    "model_id": model_id or "unknown",
                    "endpoint": endpoint or "unknown",
                },
            )
    except Exception:
        # Metrics recording should never crash the app
        pass


def record_llm_ttft(
    provider: str,
    model_id: str,
    endpoint: str,
    ttft_seconds: float,
) -> None:
    """
    Record LLM time to first token (TTFT) in seconds.

    Args:
        provider: Provider identifier (e.g., 'openai', 'google', 'vllm')
        model_id: Model identifier (e.g., 'gpt-4o', 'gemini-flash')
        endpoint: Backend endpoint (e.g., '/chat', '/chat.stream')
        ttft_seconds: Time to first token in seconds
    """
    try:
        init_metrics()

        if _llm_ttft_seconds and ttft_seconds is not None:
            _llm_ttft_seconds.record(
                ttft_seconds,
                attributes={
                    "provider": provider or "unknown",
                    "model_id": model_id or "unknown",
                    "endpoint": endpoint or "unknown",
                },
            )
    except Exception:
        # Metrics recording should never crash the app
        pass


def record_llm_tokens_per_second(
    provider: str,
    model_id: str,
    endpoint: str,
    tokens_per_second: float,
) -> None:
    """
    Record LLM tokens per second (TPS).

    Args:
        provider: Provider identifier (e.g., 'openai', 'google', 'vllm')
        model_id: Model identifier (e.g., 'gpt-4o', 'gemini-flash')
        endpoint: Backend endpoint (e.g., '/chat', '/chat.stream')
        tokens_per_second: Tokens generated per second
    """
    try:
        init_metrics()

        if _llm_tokens_per_second and tokens_per_second is not None:
            _llm_tokens_per_second.record(
                tokens_per_second,
                attributes={
                    "provider": provider or "unknown",
                    "model_id": model_id or "unknown",
                    "endpoint": endpoint or "unknown",
                },
            )
    except Exception:
        # Metrics recording should never crash the app
        pass


def record_llm_tokens(
    provider: str,
    model_id: str,
    token_type: str,
    count: int,
) -> None:
    """
    Increment the pep_llm_tokens_total counter for prompt/completion tokens.

    Args:
        provider: Provider identifier (e.g., 'openai', 'google', 'vllm')
        model_id: Model identifier (e.g., 'gpt-4o', 'gemini-flash')
        token_type: Either 'prompt' or 'completion'
        count: Number of tokens to add (ignored if falsy or <= 0)
    """
    if not count or count <= 0:
        return

    try:
        init_metrics()

        if _llm_tokens_total:
            _llm_tokens_total.add(
                count,
                attributes={
                    "provider": provider or "unknown",
                    "model_id": model_id or "unknown",
                    "type": token_type or "unknown",
                },
            )
    except Exception:
        # Metrics recording should never crash the app
        pass


def record_llm_cost(
    provider: str,
    model_id: str,
    cost_usd: float,
) -> None:
    """
    Increment the pep_llm_cost_usd_total counter.

    Args:
        provider: Provider identifier (e.g., 'openai', 'google', 'vllm')
        model_id: Model identifier (e.g., 'gpt-4o', 'gemini-flash')
        cost_usd: Cost in USD (ignored if None or <= 0)
    """
    if cost_usd is None or cost_usd <= 0:
        return

    try:
        init_metrics()

        if _llm_cost_usd_total:
            _llm_cost_usd_total.add(
                cost_usd,
                attributes={
                    "provider": provider or "unknown",
                    "model_id": model_id or "unknown",
                },
            )
    except Exception:
        # Metrics recording should never crash the app
        pass


def record_llm_tool_call(
    model_id: str,
    tool_name: str,
) -> None:
    """
    Increment the pep_llm_tool_calls_total counter.

    Args:
        model_id: Model identifier (e.g., 'gpt-4o', 'gemini-flash')
        tool_name: Name of the tool that was called
    """
    if not tool_name:
        return

    try:
        init_metrics()

        if _llm_tool_calls_total:
            _llm_tool_calls_total.add(
                1,
                attributes={
                    "model_id": model_id or "unknown",
                    "tool_name": tool_name or "unknown",
                },
            )
    except Exception:
        # Metrics recording should never crash the app
        pass


def record_runs_feedback(
    model_id: str,
    provider: str,
    feedback: int,
) -> None:
    """
    Increment the pep_runs_feedback_total counter.

    Only records feedback when value is 1 (like) or -1 (dislike).
    Skips recording for 0 (neutral) feedback.

    Args:
        model_id: Model identifier (e.g., 'gpt-4o', 'gemini-flash')
        provider: Provider identifier (e.g., 'openai', 'google', 'vllm')
        feedback: Feedback value (1 = like, -1 = dislike, 0 = neutral - not recorded)
    """
    # Only record like (1) or dislike (-1), skip neutral (0)
    if feedback not in (1, -1):
        return

    # Map numeric feedback to string labels
    feedback_map = {
        1: "like",
        -1: "dislike",
    }
    feedback_label = feedback_map.get(feedback, "unknown")

    try:
        init_metrics()

        if _runs_feedback_total:
            _runs_feedback_total.add(
                1,
                attributes={
                    "model_id": model_id or "unknown",
                    "provider": provider or "unknown",
                    "feedback": feedback_label,
                },
            )
    except Exception:
        # Metrics recording should never crash the app
        pass


def record_http_request(
    route: str,
    method: str,
    status: int,
) -> None:
    """
    Increment the pep_http_requests_total counter.

    Args:
        route: API route path (e.g., '/chat', '/prompts')
        method: HTTP method (e.g., 'GET', 'POST')
        status: HTTP status code (e.g., 200, 404, 500)
    """
    try:
        init_metrics()

        if _http_requests_total:
            _http_requests_total.add(
                1,
                attributes={
                    "route": route or "unknown",
                    "method": method or "unknown",
                    "status": str(status) if status else "unknown",
                },
            )
    except Exception:
        # Metrics recording should never crash the app
        pass
