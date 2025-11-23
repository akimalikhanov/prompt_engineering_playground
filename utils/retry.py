from typing import Callable, Optional
import logging
import uuid
import random
import time
from utils.errors import (_get_status_code, \
                    is_transient_error, \
                    prettify_openai_error, \
                    _get_retry_after_seconds)
from utils.logger_new import get_correlation_id, set_correlation_id

def compute_backoff_seconds(
    attempt: int,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter_max: float = 0.5,
    server_retry_after: Optional[float] = None,
) -> float:
    """Exponential backoff with jitter; respects Retry-After if larger."""
    backoff = min(base_delay * (2 ** (attempt - 1)), max_delay)
    jitter = random.uniform(0.0, jitter_max) if jitter_max > 0 else 0.0
    delay = backoff + jitter
    if server_retry_after:
        delay = max(delay, float(server_retry_after))
    return delay

def call_with_retry(
    fn: Callable[[], any],
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter_max: float = 0.5,
    logger: Optional[logging.Logger] = None,
    corr_id: Optional[str] = None,
):
    """
    Wrap a call with transient-error retries. Returns fn() result or raises the original error.
    """
    # Reuse request correlation id if available
    corr_id = corr_id or get_correlation_id()
    if corr_id == "-":  # no context set (e.g. background job)
        corr_id = str(uuid.uuid4())
        set_correlation_id(corr_id)
    log = logger or logging.getLogger("retry")
    attempt = 1
    while True:
        try:
            result = fn()
            # Successful call
            log.debug(f"[corr_id={corr_id}] Success on attempt {attempt}")
            return result
        except Exception as err:
            status_code = _get_status_code(err)
            transient = is_transient_error(err)
            pretty = prettify_openai_error(err)

            log.warning(
                f"[corr_id={corr_id}] API call failed (attempt {attempt}) "
                f"status={status_code} transient={transient} msg={pretty}"
            )

            if not transient or attempt > max_retries:
                # Give up
                raise

            # Compute delay (respect Retry-After)
            retry_after = _get_retry_after_seconds(err)
            sleep_s = compute_backoff_seconds(
                attempt=attempt,
                base_delay=base_delay,
                max_delay=max_delay,
                jitter_max=jitter_max,
                server_retry_after=retry_after,
            )
            log.info(f"[corr_id={corr_id}] Retrying in {sleep_s:.2f}s ...")
            time.sleep(sleep_s)
            attempt += 1