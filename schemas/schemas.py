from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Literal, Optional, Union, Dict, Any
from typing_extensions import Annotated


class JsonSchemaConfig(BaseModel):
    """Configuration for json_schema response format."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    name: Optional[str] = Field(None, description="Optional name for the schema")
    schema: Dict[str, Any] = Field(..., description="The JSON schema definition")
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


ResponseFormat = Union[ResponseFormatJsonSchema, ResponseFormatJsonObject]


class ToolFunction(BaseModel):
    """Function definition used in tools (OpenAI-style)."""

    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, Field(min_length=1, max_length=100)]
    description: Optional[str] = None
    parameters: Dict[str, Any]


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
    models: List[str]

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(min_length=1, max_length=7000)]

class ChatParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: Optional[Annotated[float, Field(ge=0.0, le=2.0)]] = 0.7
    top_p: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = 1.0
    max_tokens: Optional[Annotated[int, Field(ge=1, le=4000)]] = 512
    seed: Optional[Annotated[int, Field(ge=0, le=2_147_483_647)]] = None
    response_format: Optional[ResponseFormat] = None
    # Optional tool / function-calling configuration (OpenAI-style)
    tools: Optional[List[ToolDefinition]] = None
    # 'auto' / 'none' or an OpenAI-style dict for function choice
    tool_choice: Optional[Union[Literal["auto", "none"], Dict[str, Any]]] = None

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_id: Annotated[str, Field(min_length=1, max_length=100)]
    model_id: Annotated[str, Field(min_length=1, max_length=100)]
    messages: Union[str, List[ChatMessage]]
    params: ChatParams = ChatParams()
    context_prompt: Optional[Annotated[str, Field(min_length=1, max_length=10000)]] = None
    session_id: Optional[str] = None

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
    ttft_ms: Optional[float] = None
    latency_ms: Annotated[float, Field(ge=0)]
    model: Annotated[str, Field(min_length=1, max_length=100)]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None
    cost_usd: Optional[float] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    executed_tools: Optional[List[str]] = None

class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Annotated[str, Field(min_length=0, max_length=7000)]
    metrics: Metrics
    tool_messages: Optional[List[str]] = None  # Formatted tool execution messages from backend

class RunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    occurred_at: str
    trace_id: str
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    provider_key: str
    model_id: str
    params_json: Dict[str, Any] = {}
    input_text: Optional[str] = None
    system_prompt: Optional[str] = None
    context_prompt: Optional[str] = None
    output_text: Optional[str] = None
    output_preview: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None
    ttft_ms: Optional[int] = None
    status: str
    error_type: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    cached: bool = False


class RunFeedbackRequest(BaseModel):
    """Payload for updating user feedback on a run, looked up by trace_id."""

    model_config = ConfigDict(extra="forbid")
    trace_id: Annotated[str, Field(min_length=1, max_length=200)]
    # -1 = negative, 0 = neutral/cleared, 1 = positive
    user_feedback: Annotated[int, Field(ge=-1, le=1)]
