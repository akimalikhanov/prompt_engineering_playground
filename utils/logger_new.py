import json
import logging
import os
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Dict, Optional, Union

from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider as OtelLoggerProvider, LoggingHandler as OtelLoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

# Hold references so the GC doesn't tear down OTEL processors
_OTEL_RUNTIME_REFS = []
# Correlation ID (request-scoped)
_corr_id_var: ContextVar[str] = ContextVar("corr_id", default="-")

def get_correlation_id() -> str:
    return _corr_id_var.get()

def set_correlation_id(corr_id: Optional[str] = None) -> str:
    if not corr_id:
        corr_id = str(uuid.uuid4())
    _corr_id_var.set(corr_id)
    return corr_id

class CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True

THIRD_PARTY_LOGGERS = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "gunicorn",
    "gunicorn.error",
    "gunicorn.access",
    "httpx",
    "httpx._client",
    "httpx._transports.default",
    "httpcore",
)

class JsonLogFormatter(logging.Formatter):
    """Emit structured JSON logs that encode useful metadata."""

    RESERVED_ATTRS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
    }

    def __init__(self, *, default_event: str = "app_log"):
        super().__init__()
        self._default_event = default_event

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self._format_ts(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "correlation_id", "-"),
            "event": getattr(record, "event", self._default_event),
        }
        payload.update(self._extract_extra(record))

        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _format_ts(created: float) -> str:
        dt = datetime.fromtimestamp(created, tz=timezone.utc)
        return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _extract_extra(self, record: logging.LogRecord) -> Dict[str, Any]:
        extras: Dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key in self.RESERVED_ATTRS or key.startswith("_"):
                continue
            if key in {"correlation_id", "trace_id", "event", "message"}:
                continue
            extras[key] = self._serialize(value)
        return extras

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {k: self._serialize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialize(v) for v in value]
        if hasattr(value, "value"):
            # simple support for Enum values
            return getattr(value, "value")
        return repr(value)

def _ensure_logs_dir(path: str):
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    # TimedRotatingFileHandler expects the target file to exist when it attempts
    # to rotate (it renames the current file). Ensure the file is present so the
    # first rotation doesn't raise FileNotFoundError.
    if not os.path.exists(path):
        open(path, "a", encoding="utf-8").close()

def _add_handler_once(logger: logging.Logger, handler: logging.Handler):
    # Avoid duplicate handlers when re-calling setup
    sig = (handler.__class__.__name__, getattr(handler, "baseFilename", None))
    for h in logger.handlers:
        if h.__class__.__name__ == sig[0] and getattr(h, "baseFilename", None) == sig[1]:
            return
    logger.addHandler(handler)

def _hijack_loggers(logger_names, level: int):
    for name in logger_names:
        target = logging.getLogger(name)
        target.setLevel(level)
        target.handlers = []
        target.propagate = True


def _parse_otel_headers(headers: Union[str, Dict[str, str], None]) -> Optional[Dict[str, str]]:
    if headers is None:
        env_headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
        headers = env_headers
    if isinstance(headers, dict):
        return headers
    if not headers:
        return None
    parsed: Dict[str, str] = {}
    for pair in headers.split(","):
        if not pair:
            continue
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed or None


def _build_otel_handler(
    *,
    level: int,
    corr_filter: logging.Filter,
    service_name: str,
    endpoint: Optional[str],
    headers: Union[str, Dict[str, str], None],
) -> Optional[logging.Handler]:
    resolved_endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not resolved_endpoint:
        resolved_endpoint = "http://localhost:4318/v1/logs"
    resolved_endpoint = resolved_endpoint.rstrip("/")
    if not resolved_endpoint.endswith("/v1/logs"):
        resolved_endpoint = f"{resolved_endpoint}/v1/logs"

    header_dict = _parse_otel_headers(headers)

    resource = Resource.create({"service.name": service_name})
    provider = OtelLoggerProvider(resource=resource)
    exporter = OTLPLogExporter(endpoint=resolved_endpoint, headers=header_dict)
    processor = BatchLogRecordProcessor(exporter)
    provider.add_log_record_processor(processor)

    handler = OtelLoggingHandler(level=level, logger_provider=provider)
    handler.addFilter(corr_filter)
    _OTEL_RUNTIME_REFS.append((provider, processor, handler))
    return handler

def setup_logging(
    *,
    app_logger_name: str = "app",
    level: int = logging.INFO,
    to_console: bool = True,
    to_file: bool = True,
    file_path: str = "logs/app.log",
    when: str = "midnight",   # Timed rotating (midnight rollover)
    backup_count: int = 7,     # keep last 7 files
    hijack_uvicorn: bool = True,
    enable_otel: bool = False,
    otel_endpoint: Optional[str] = None,
    otel_headers: Union[str, Dict[str, str], None] = None,
) -> logging.Logger:
    """
    Idempotent logging setup. Returns the named app logger.
    - Adds correlation_id to every record via a filter.
    - Optionally hijacks Uvicorn loggers to reuse our handlers (prevents duplicates).
    - Optionally emits logs to an OTLP/OTel collector.
    """
    # Root level (affects all unless propagate=False)
    logging.basicConfig(level=level)
    root = logging.getLogger()
    root.setLevel(level)

    formatter = JsonLogFormatter()
    corr_filter = CorrelationFilter()

    logger = logging.getLogger(app_logger_name)
    logger.setLevel(level)
    logger.propagate = False  # prevent duplicate printing via root

    if to_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        ch.addFilter(corr_filter)
        _add_handler_once(logger, ch)

    if to_file:
        _ensure_logs_dir(file_path)
        fh = TimedRotatingFileHandler(file_path, when=when, backupCount=backup_count, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        fh.addFilter(corr_filter)
        _add_handler_once(logger, fh)

    if enable_otel:
        otel_handler = _build_otel_handler(
            level=level,
            corr_filter=corr_filter,
            service_name=os.getenv("OTEL_SERVICE_NAME", app_logger_name),
            endpoint=otel_endpoint,
            headers=otel_headers,
        )
        if otel_handler:
            _add_handler_once(logger, otel_handler)

    if hijack_uvicorn:
        # Make uvicorn/gunicorn/httpx loggers reuse our handlers & avoid double output
        _hijack_loggers(THIRD_PARTY_LOGGERS, level)
        # Replace root handlers with our JSON ones so propagated logs share format
        root.handlers = []
        for h in logger.handlers:
            _add_handler_once(root, h)
        # Optionally, keep high-volume access logs quieter
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("gunicorn.access").setLevel(logging.WARNING)

    logging.getLogger("watchfiles").setLevel(logging.WARNING)

    return logger

# ---------- FastAPI middleware helpers ----------

async def correlation_id_middleware(request, call_next):
    """
    Starlette/FastAPI-compatible middleware function.
    Pulls correlation id from request or creates one. Adds it to response header.
    Usage:
        app.middleware("http")(correlation_id_middleware)
    """
    incoming = request.headers.get("x-request-id") or request.headers.get("x-correlation-id")
    corr_id = set_correlation_id(incoming)
    try:
        response = await call_next(request)
    finally:
        # Ensure the corr id remains per-request; no explicit reset needed as contextvar is scoped
        pass
    response.headers["X-Request-ID"] = corr_id
    return response


def log_llm_call(
    *,
    logger: Optional[logging.Logger] = None,
    logger_name: str = "app",
    event: str = "llm_call_completed",
    level: int = logging.INFO,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    request_type: Optional[str] = None,
    api_endpoint: Optional[str] = None,
    backend_endpoint: Optional[str] = None,
    http_status: Optional[int] = None,
    status: Optional[str] = None,
    latency_ms: Optional[float] = None,
    ttft_ms: Optional[float] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
    error_msg: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit a structured log describing an LLM provider API call with consistent fields.
    Use this helper wherever httpx/provider calls occur so terminal & file logs match.
    """
    log = logger or logging.getLogger(logger_name)
    resolved_trace_id = trace_id or get_correlation_id()
    normalized_request_type = request_type.upper() if isinstance(request_type, str) else request_type

    payload: Dict[str, Any] = {
        "event": event,
        "trace_id": resolved_trace_id,
        "session_id": session_id,
        "provider": provider,
        "model_id": model_id,
        "request_type": normalized_request_type,
        "api_endpoint": api_endpoint,
        "backend_endpoint": backend_endpoint,
        "http_status": http_status,
        "status": status,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost_usd,
        "error_msg": error_msg,
    }

    if extra_fields:
        payload.update(extra_fields)

    log.log(level, event, extra=payload)