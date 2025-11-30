"""
OpenTelemetry Metrics instrumentation for the LLM playground.

Exports metrics via OTLP to the collector, which then exposes them to Prometheus.
"""
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import Counter

from utils.otel_config import get_otel_resource, get_otlp_metrics_endpoint

# ---------------------------------------------------------------------------
# Lazy initialization (singleton pattern)
# ---------------------------------------------------------------------------
_initialized: bool = False
_llm_requests_total: Counter | None = None


def _ensure_initialized() -> None:
    """Initialize MeterProvider once, on first use."""
    global _initialized, _llm_requests_total
    
    if _initialized:
        return
    
    # Check if a MeterProvider is already set (e.g., by another library)
    current_provider = metrics.get_meter_provider()
    is_noop = type(current_provider).__name__ == "NoOpMeterProvider"
    
    if is_noop:
        # No provider set yet, create and set ours
        exporter = OTLPMetricExporter(endpoint=get_otlp_metrics_endpoint())
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=15000)
        provider = MeterProvider(resource=get_otel_resource(), metric_readers=[reader])
        metrics.set_meter_provider(provider)
    
    # Get meter (works with either our provider or an existing one)
    meter = metrics.get_meter("pep.llm", version="0.1.0")
    
    # Create metrics
    _llm_requests_total = meter.create_counter(
        name="pep_llm_requests_total",
        description="Total number of LLM API requests",
        unit="1",
    )
    
    _initialized = True


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
    _ensure_initialized()
    
    _llm_requests_total.add(
        1,
        attributes={
            "provider": provider or "unknown",
            "model_id": model_id or "unknown",
            "status": status or "unknown",
            "endpoint": endpoint or "unknown",
        },
    )
