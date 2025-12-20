from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JsonSchemaConfig(BaseModel):
    """Configuration for json_schema response format."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    name: str | None = Field(None, description="Optional name for the schema")
    schema: dict[str, Any] = Field(..., description="The JSON schema definition")  # type: ignore[reportIncompatibleMethodOverride]
    strict: bool = True


class ResponseFormatJsonSchema(BaseModel):
    """Response format for json_schema type."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["json_schema"] = "json_schema"
    json_schema: JsonSchemaConfig


class ResponseFormatJsonObject(BaseModel):
    """Response format for json_object type."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["json_object"] = "json_object"


ResponseFormat = ResponseFormatJsonSchema | ResponseFormatJsonObject


class ToolFunction(BaseModel):
    """Function definition used in tools (OpenAI-style)."""

    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, Field(min_length=1, max_length=100)]
    description: str | None = None
    parameters: dict[str, Any]


class ToolDefinition(BaseModel):
    """Tool definition, currently limited to function tools."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["function"] = "function"
    function: ToolFunction


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"] = "ok"


class RootResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"] = "ok"
    message: Annotated[str, Field(min_length=1, max_length=100)]


class ModelsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[str]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(min_length=1, max_length=7000)]


class ChatParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] | None = None
    top_p: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    max_tokens: Annotated[int, Field(ge=1, le=4000)] | None = None
    seed: Annotated[int, Field(ge=0, le=2147483647)] | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    response_format: ResponseFormat | None = None
    # Optional tool / function-calling configuration (OpenAI-style)
    tools: list[ToolDefinition] | None = None
    # 'auto' / 'none' or an OpenAI-style dict for function choice
    tool_choice: Literal["auto", "none"] | dict[str, Any] | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_id: Annotated[str, Field(min_length=1, max_length=100)]
    model_id: Annotated[str, Field(min_length=1, max_length=100)]
    messages: str | list[ChatMessage]
    params: ChatParams = ChatParams()
    context_prompt: Annotated[str, Field(min_length=1, max_length=10000)] | None = None
    session_id: str | None = None

    @field_validator("provider_id", "model_id")
    @classmethod
    def strip_nonempty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("must not be empty")
        return v2

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        # Accept non-empty string or a non-empty list of ChatMessage
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("messages string must not be empty")
            return v
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("messages list must contain at least one item")
            return v
        raise ValueError("messages must be a string or a list of ChatMessage")


class Usage(BaseModel):
    prompt_tokens: Annotated[int, Field(ge=0)]
    completion_tokens: Annotated[int, Field(ge=0)]
    total_tokens: Annotated[int, Field(ge=0)]


class Metrics(BaseModel):
    ttft_ms: float | None = None
    latency_ms: Annotated[float, Field(ge=0)]
    model: Annotated[str, Field(min_length=1, max_length=100)]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    tokens_per_second: float | None = None
    cost_usd: float | None = None
    session_id: str | None = None
    trace_id: str | None = None
    executed_tools: list[str] | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Annotated[str, Field(min_length=0, max_length=7000)]
    metrics: Metrics
    tool_messages: list[str] | None = None  # Formatted tool execution messages from backend


class RunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    occurred_at: str
    trace_id: str
    request_id: str | None = None
    session_id: str | None = None
    provider_key: str
    model_id: str
    params_json: dict[str, Any] = {}
    input_text: str | None = None
    system_prompt: str | None = None
    context_prompt: str | None = None
    output_text: str | None = None
    output_preview: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    ttft_ms: int | None = None
    status: str
    error_type: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    cached: bool = False


class RunFeedbackRequest(BaseModel):
    """Payload for updating user feedback on a run, looked up by trace_id."""

    model_config = ConfigDict(extra="forbid")
    trace_id: Annotated[str, Field(min_length=1, max_length=200)]
    # -1 = negative, 0 = neutral/cleared, 1 = positive
    user_feedback: Annotated[int, Field(ge=-1, le=1)]
