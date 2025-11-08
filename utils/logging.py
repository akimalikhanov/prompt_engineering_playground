import logging
import os
import uuid
from logging.handlers import TimedRotatingFileHandler
from typing import Optional
from contextvars import ContextVar

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

DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s [corr_id=%(correlation_id)s] - %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

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
) -> logging.Logger:
    """
    Idempotent logging setup. Returns the named app logger.
    - Adds correlation_id to every record via a filter.
    - Optionally hijacks Uvicorn loggers to reuse our handlers (prevents duplicates).
    """
    # Root level (affects all unless propagate=False)
    logging.basicConfig(level=level)

    formatter = logging.Formatter(fmt=DEFAULT_FMT, datefmt=DEFAULT_DATEFMT)
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

    if hijack_uvicorn:
        # Make uvicorn loggers reuse our handlers & avoid double output
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            uvlog = logging.getLogger(name)
            uvlog.setLevel(level)
            uvlog.handlers = []           # clear default handlers
            uvlog.propagate = True        # bubble up to root OR to our app logger
        # Option A: attach our handlers to ROOT so uvicorn logs share them
        root = logging.getLogger()
        root.setLevel(level)
        # attach same handlers as our app logger (once)
        for h in logger.handlers:
            _add_handler_once(root, h)
        # Optionally, keep uvicorn.access a bit quieter:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

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