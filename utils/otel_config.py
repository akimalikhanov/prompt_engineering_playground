"""
Unified OpenTelemetry configuration for the LLM playground.

This module provides a single source of truth for OTEL resource attributes
used across logs, metrics, and traces.
"""
import os
import uuid
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_NAMESPACE, DEPLOYMENT_ENVIRONMENT


# ---------------------------------------------------------------------------
# Process instance ID (unique per process, persists for process lifetime)
# ---------------------------------------------------------------------------
# This is critical for multi-worker setups (e.g., Gunicorn with multiple workers).
# Each worker process needs a unique identifier so metrics from different
# workers don't overwrite each other in the OTEL collector/Prometheus.
_PROCESS_INSTANCE_ID: str = str(uuid.uuid4())[:8]  # Short UUID for readability


# ---------------------------------------------------------------------------
# Configuration (read from environment with sensible defaults)
# ---------------------------------------------------------------------------
def get_service_name() -> str:
    return os.getenv("OTEL_SERVICE_NAME", "pep-api")


def get_service_namespace() -> str:
    return os.getenv("OTEL_SERVICE_NAMESPACE", "pep")


def get_deployment_environment() -> str:
    return os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT", "prod")


def get_otlp_endpoint() -> str:
    """Get base OTLP endpoint (without signal-specific path)."""
    return os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")


def get_otlp_signal_endpoint(signal: str) -> str:
    """
    Get OTLP endpoint for a specific signal type.
    
    Args:
        signal: One of 'metrics', 'logs', or 'traces'
    
    Returns:
        Full endpoint URL with /v1/{signal} path appended if needed.
    """
    env_var = f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT"
    path_suffix = f"/v1/{signal}"
    
    base = os.getenv(env_var, get_otlp_endpoint()).rstrip("/")
    if not base.endswith(path_suffix):
        return f"{base}{path_suffix}"
    return base


# Convenience aliases for backward compatibility
def get_otlp_metrics_endpoint() -> str:
    return get_otlp_signal_endpoint("metrics")


def get_otlp_logs_endpoint() -> str:
    return get_otlp_signal_endpoint("logs")


def get_otlp_traces_endpoint() -> str:
    return get_otlp_signal_endpoint("traces")


# ---------------------------------------------------------------------------
# Shared Resource (singleton)
# ---------------------------------------------------------------------------
_resource: Resource | None = None


def get_process_instance_id() -> str:
    """Get the unique process instance ID for this worker."""
    return _PROCESS_INSTANCE_ID


def get_otel_resource() -> Resource:
    """
    Get the shared OTEL Resource with service identity attributes.
    
    This resource should be used by all OTEL providers (logs, metrics, traces)
    to ensure consistent service identification across all signals.
    
    Note: Includes a unique process instance ID to differentiate metrics
    from different worker processes in multi-worker setups (Gunicorn, etc.).
    """
    global _resource
    if _resource is None:
        _resource = Resource.create({
            SERVICE_NAME: get_service_name(),
            SERVICE_NAMESPACE: get_service_namespace(),
            DEPLOYMENT_ENVIRONMENT: get_deployment_environment(),
            # "service.instance.id": get_process_instance_id(),
        })
    return _resource


def get_resource_attributes_string(include_instance_id: bool = False) -> str:
    """
    Get resource attributes as a comma-separated string for env vars.
    
    Format (example): "service.name=X,service.namespace=Y,deployment.environment=Z"
    Optionally include "service.instance.id" when include_instance_id is True.
    """
    attrs = [
        f"service.name={get_service_name()}",
        f"service.namespace={get_service_namespace()}",
        f"deployment.environment={get_deployment_environment()}",
    ]
    if include_instance_id:
        attrs.append(f"service.instance.id={get_process_instance_id()}")
    return ",".join(attrs)


def configure_otel_env_vars() -> None:
    """
    Set OTEL environment variables for libraries that read them directly.
    
    Call this early in application startup (before importing libraries
    that auto-configure from env vars like MLflow).
    """
    service_name = get_service_name()
    
    # Service identity
    os.environ.setdefault("OTEL_SERVICE_NAME", service_name)
    os.environ.setdefault("OTEL_RESOURCE_ATTRIBUTES", get_resource_attributes_string())
    
    # OTLP protocol
    os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
    
    # Endpoints
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", get_otlp_endpoint())
    os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", get_otlp_traces_endpoint())
    os.environ.setdefault("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", get_otlp_logs_endpoint())
    os.environ.setdefault("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", get_otlp_metrics_endpoint())

