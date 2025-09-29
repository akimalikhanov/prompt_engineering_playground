from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Literal, Optional, Union
from typing_extensions import Annotated


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

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_id: Annotated[str, Field(min_length=1, max_length=100)]
    model_id: Annotated[str, Field(min_length=1, max_length=100)]
    messages: Union[str, List[ChatMessage]]
    params: ChatParams = ChatParams()

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

class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Annotated[str, Field(min_length=0, max_length=7000)]
    usage: Usage
    metrics: Metrics
