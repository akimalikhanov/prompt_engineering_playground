"""Tests for API error handling, retry logic, and SSE utilities."""

from unittest.mock import Mock

import openai
import pytest

from utils.errors import (
    BackendError,
    _parse_error_text,
    is_rate_limit,
    is_transient_error,
    prettify_openai_error,
)
from utils.retry import compute_backoff_seconds
from utils.sse import sse_pack


class _MockAPIError(Exception):
    """Mock API error with status_code and response attributes for testing."""

    def __init__(self, message: str, status_code: int | None = None, response: Mock | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class TestTransientErrorClassification:
    """Transient vs permanent error classification for retry decisions."""

    @pytest.mark.parametrize(
        "status_code,is_transient",
        [
            (429, True),  # Rate limit
            (500, True),  # Server error
            (502, True),  # Bad gateway
            (503, True),  # Service unavailable
            (504, True),  # Gateway timeout
            (408, True),  # Request timeout
            (400, False),  # Bad request - client error
            (401, False),  # Unauthorized
            (403, False),  # Forbidden
            (404, False),  # Not found
            (422, False),  # Validation error
        ],
    )
    def test_status_code_classification(self, status_code, is_transient):
        """HTTP status codes are classified correctly."""
        error = _MockAPIError("test", status_code=status_code)
        assert is_transient_error(error) == is_transient

    def test_network_errors_are_transient(self):
        """Network connection errors are transient."""
        error = openai.APIConnectionError(message="Connection failed", request=Mock())
        assert is_transient_error(error) is True

    def test_timeout_errors_are_transient(self):
        """Timeout errors are transient."""
        error = openai.APITimeoutError(request=Mock())
        assert is_transient_error(error) is True


class TestRateLimitDetection:
    """Rate limit detection for special handling."""

    def test_detects_429_status(self):
        """429 status code is recognized as rate limit."""
        error = _MockAPIError("Rate limited", status_code=429)
        assert is_rate_limit(error) is True

    def test_detects_rate_limit_error_type(self):
        """OpenAI RateLimitError type is recognized."""
        if hasattr(openai, "RateLimitError"):
            error = openai.RateLimitError(message="Rate limited", response=Mock(), body=None)
            assert is_rate_limit(error) is True


class TestErrorPrettification:
    """Human-readable error message generation."""

    def test_network_error_message(self):
        """Network errors produce helpful messages."""
        error = openai.APIConnectionError(message="Connection failed", request=Mock())
        result = prettify_openai_error(error)
        assert "Network" in result

    def test_timeout_error_message(self):
        """Timeout errors produce helpful messages."""
        error = openai.APITimeoutError(request=Mock())
        result = prettify_openai_error(error)
        assert "Timeout" in result

    def test_status_code_in_message(self):
        """HTTP status codes are included in error messages."""
        error = _MockAPIError("Error", status_code=404)
        result = prettify_openai_error(error)
        assert "404" in result

    def test_api_detail_extracted(self):
        """API error details are extracted from response JSON."""
        response = Mock()
        response.json.return_value = {"error": {"message": "Invalid model"}}
        error = _MockAPIError("Error", status_code=400, response=response)

        result = prettify_openai_error(error)
        assert "Invalid model" in result


class TestErrorTextParsing:
    """Parsing error bodies from API responses."""

    def test_parses_json_error(self):
        """JSON error body is parsed correctly."""
        status, message = _parse_error_text('{"error": "Bad request", "status": 400}')
        assert status == 400
        assert message == "Bad request"

    def test_plain_text_fallback(self):
        """Non-JSON error text is returned as-is."""
        status, message = _parse_error_text("Plain error message")
        assert status is None
        assert message == "Plain error message"

    def test_empty_text_handled(self):
        """Empty error text produces None values."""
        status, message = _parse_error_text("")
        assert status is None
        assert message is None


class TestBackendError:
    """Custom BackendError exception behavior."""

    def test_carries_status_code(self):
        """BackendError carries message and status code."""
        error = BackendError("Something went wrong", 500)
        assert error.message == "Something went wrong"
        assert error.status_code == 500
        assert str(error) == "Something went wrong"


class TestExponentialBackoff:
    """Retry backoff timing calculation."""

    def test_exponential_growth(self):
        """Backoff increases exponentially with attempts."""
        delays = [
            compute_backoff_seconds(i, base_delay=1.0, max_delay=100.0, jitter_max=0.0)
            for i in range(1, 5)
        ]
        # Delays: 1.0, 2.0, 4.0, 8.0
        assert delays == [1.0, 2.0, 4.0, 8.0]

    def test_respects_max_delay(self):
        """Backoff is capped at max_delay."""
        delay = compute_backoff_seconds(100, base_delay=1.0, max_delay=10.0, jitter_max=0.0)
        assert delay == 10.0

    def test_jitter_adds_randomness(self):
        """Jitter adds random variation to backoff."""
        delays = [
            compute_backoff_seconds(1, base_delay=1.0, max_delay=10.0, jitter_max=1.0)
            for _ in range(10)
        ]
        # With jitter, we should get varying delays between 1.0 and 2.0
        assert min(delays) >= 1.0
        assert max(delays) <= 2.0
        # Should have some variation (not all identical)
        assert len(set(delays)) > 1

    def test_respects_server_retry_after(self):
        """Server-provided Retry-After is honored when larger."""
        delay = compute_backoff_seconds(
            1, base_delay=1.0, max_delay=10.0, jitter_max=0.0, server_retry_after=30.0
        )
        assert delay == 30.0


class TestSSEPacking:
    """Server-Sent Events message formatting."""

    def test_basic_data_message(self):
        """Simple data is packed correctly."""
        result = sse_pack("Hello world").decode("utf-8")
        assert "data: Hello world" in result
        assert result.endswith("\n\n")

    def test_multiline_data(self):
        """Multiline data produces multiple data: lines."""
        result = sse_pack("Line 1\nLine 2").decode("utf-8")
        assert "data: Line 1" in result
        assert "data: Line 2" in result

    def test_full_message_with_metadata(self):
        """Event, id, and retry are all included when provided."""
        result = sse_pack("msg", event="update", id="123", retry=5000).decode("utf-8")
        assert "event: update" in result
        assert "id: 123" in result
        assert "retry: 5000" in result
        assert "data: msg" in result

    def test_utf8_encoding(self):
        """Non-ASCII characters are encoded properly."""
        result = sse_pack("Hello 世界 🌍").decode("utf-8")
        assert "Hello 世界 🌍" in result
