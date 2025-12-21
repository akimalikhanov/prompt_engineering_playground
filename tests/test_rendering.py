"""Tests for template rendering and message construction."""

import pytest

from utils.jinja_renderer import extract_variables_from_template, render_messages, render_template
from utils.message_utils import _build_messages, _build_response_format, _coerce_seed


class TestJinjaTemplateRendering:
    """Core Jinja2 template rendering behavior."""

    @pytest.mark.parametrize(
        "template,variables,expected",
        [
            ("Hello {{ name }}", {"name": "Alice"}, "Hello Alice"),
            ("{{ a }} + {{ b }} = {{ c }}", {"a": 1, "b": 2, "c": 3}, "1 + 2 = 3"),
            ("{{ name | upper }}", {"name": "bob"}, "BOB"),
            ("No variables here", {}, "No variables here"),
            ("", {}, ""),
        ],
    )
    def test_basic_rendering(self, template, variables, expected):
        """Template rendering with various inputs."""
        rendered, missing, _ = render_template(template, variables)
        assert rendered == expected
        assert missing == []

    def test_missing_variable_reported(self):
        """Missing template variables are detected and reported."""
        rendered, missing, warnings = render_template("Hello {{ name }}", {})
        assert "name" in missing
        assert len(warnings) > 0

    def test_unused_variable_warned(self):
        """Unused variables generate warnings."""
        _, _, warnings = render_template("Hello {{ name }}", {"name": "X", "unused": "Y"})
        assert any("Unused" in w for w in warnings)

    def test_syntax_error_handled(self):
        """Invalid Jinja syntax doesn't crash, produces warning."""
        rendered, _, warnings = render_template("Hello {{ name }", {"name": "X"})
        assert any("syntax" in w.lower() for w in warnings)
        # Returns original template on syntax error
        assert rendered == "Hello {{ name }"


class TestVariableExtraction:
    """Variable extraction from templates."""

    @pytest.mark.parametrize(
        "template,expected",
        [
            ("{{ name }}", ["name"]),
            ("{{ a }} {{ b }}", ["a", "b"]),
            ("{{ name | upper }}", ["name"]),
            ("{{ x }} and {{ x }}", ["x"]),  # Duplicates removed
            ("No vars", []),
        ],
    )
    def test_extraction(self, template, expected):
        """Variables are correctly extracted from templates."""
        assert extract_variables_from_template(template) == sorted(expected)


class TestMessageRendering:
    """Multi-message Jinja rendering."""

    def test_renders_all_messages(self):
        """All messages in a list are rendered with variables."""
        messages = [
            {"role": "system", "content": "You are {{ bot_name }}"},
            {"role": "user", "content": "Hello, I'm {{ user_name }}"},
        ]
        rendered, missing, _ = render_messages(messages, {"bot_name": "Hal", "user_name": "Dave"})

        assert len(rendered) == 2
        assert rendered[0]["content"] == "You are Hal"
        assert rendered[1]["content"] == "Hello, I'm Dave"
        assert missing == []

    def test_missing_var_across_messages(self):
        """Missing variables are aggregated across all messages."""
        messages = [
            {"role": "user", "content": "{{ a }} and {{ b }}"},
            {"role": "user", "content": "{{ c }}"},
        ]
        _, missing, _ = render_messages(messages, {"a": "1"})
        assert set(missing) == {"b", "c"}


class TestMessageBuilding:
    """Message structure construction for API calls."""

    def test_user_message_only(self):
        """Minimal message with just user content."""
        messages = _build_messages("Hello", [], None, None)
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_with_system_prompt(self):
        """System prompt is prepended correctly."""
        messages = _build_messages("Hello", [], "Be helpful", None)
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "Be helpful"}
        assert messages[1]["role"] == "user"

    def test_with_history(self):
        """Conversation history is preserved in order."""
        history = [
            {"role": "user", "content": "Previous"},
            {"role": "assistant", "content": "Response"},
        ]
        messages = _build_messages("New message", history, None, None)

        assert len(messages) == 3
        assert messages[0]["content"] == "Previous"
        assert messages[1]["content"] == "Response"
        assert messages[2]["content"] == "New message"

    def test_full_message_construction(self):
        """All components combined in correct order."""
        messages = _build_messages(
            "Question",
            [{"role": "user", "content": "Prior"}],
            "System",
            "Context info",
        )
        # Order: system, context, history, new message
        assert messages[0]["role"] == "system"
        assert "CONTEXT:" in messages[1]["content"]
        assert messages[2]["content"] == "Prior"
        assert messages[3]["content"] == "Question"


class TestSeedCoercion:
    """Seed value normalization for reproducibility."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            (None, None),
            ("42", 42),
            (42, 42),
            (42.9, 42),
            ("-5", 0),  # Negative clamped to 0
            (-5, 0),
            ("", None),
            ("   ", None),
            ("invalid", None),
            (0, 0),
        ],
    )
    def test_seed_normalization(self, input_val, expected):
        """Various inputs are normalized to valid seed or None."""
        assert _coerce_seed(input_val) == expected


class TestResponseFormat:
    """JSON response format configuration."""

    def test_none_mode_returns_none(self):
        """'None' mode produces no response format."""
        assert _build_response_format("None", False, None) is None

    def test_json_object_mode(self):
        """JSON object mode produces correct format."""
        result = _build_response_format("JSON object", False, None)
        assert result == {"type": "json_object"}

    def test_json_schema_mode(self):
        """JSON schema mode parses and wraps schema correctly."""
        schema = '{"type": "object", "properties": {"name": {"type": "string"}}}'
        result = _build_response_format("JSON schema", True, schema)

        assert result is not None
        assert result["type"] == "json_schema"
        assert result["json_schema"]["strict"] is True
        assert result["json_schema"]["schema"]["type"] == "object"

    def test_invalid_json_schema_returns_none(self):
        """Invalid JSON schema returns None instead of crashing."""
        result = _build_response_format("JSON schema", True, "not valid json")
        assert result is None
