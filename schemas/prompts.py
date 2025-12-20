"""
Pydantic schemas for prompt management endpoints
"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ============================================
# Message schema (for prompt messages)
# ============================================
class PromptMessage(BaseModel):
    """Single message in a prompt template"""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


# ============================================
# Variable schema (for prompt variables)
# ============================================
class PromptVariable(BaseModel):
    """Variable definition for prompt templates"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    type: str = "string"  # string, number, boolean, etc.
    required: bool = True
    desc: str | None = None


# ============================================
# Request schemas
# ============================================
class CreatePromptRequest(BaseModel):
    """Request to create a new prompt (v1)"""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    technique: str = Field(min_length=1)
    prompt_template: list[PromptMessage] = Field(min_length=1)
    description: str | None = None
    category: str | None = None
    tags: list[str] | None = None
    variables: list[PromptVariable] | None = []
    default_examples: list[dict[str, Any]] | None = None
    response_format: str | None = None
    json_schema_template: dict[str, Any] | None = None
    tool_config: dict[str, Any] | None = None


class CreateVersionRequest(BaseModel):
    """Request to create a new version of an existing prompt"""

    model_config = ConfigDict(extra="forbid")

    prompt_template: list[PromptMessage] | None = None
    variables: list[PromptVariable] | None = None
    description: str | None = None
    category: str | None = None
    tags: list[str] | None = None
    default_examples: list[dict[str, Any]] | None = None
    response_format: str | None = None
    json_schema_template: dict[str, Any] | None = None
    tool_config: dict[str, Any] | None = None


class RenderPromptRequest(BaseModel):
    """Request to render a prompt with variables"""

    model_config = ConfigDict(extra="forbid")

    variables: dict[str, Any] = Field(default_factory=dict)


class PatchPromptRequest(BaseModel):
    """Request to update prompt status/enabled flag"""

    model_config = ConfigDict(extra="forbid")

    is_active: bool | None = None


# ============================================
# Response schemas
# ============================================
class PromptExampleResponse(BaseModel):
    """Response schema for a prompt example"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    key: str
    title: str
    version: int
    technique: str
    prompt_template: list[
        dict[str, str]
    ]  # Messages array: [{"role": "system", "content": "..."}, ...]
    variables: list[dict[str, Any]]  # Variable definitions
    is_active: bool
    description: str | None = None
    category: str | None = None
    tags: list[str] | None = None
    default_examples: list[dict[str, Any]] | None = None
    response_format: str | None = None
    json_schema_template: dict[str, Any] | None = None
    tool_config: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class ListPromptsResponse(BaseModel):
    """Response schema for listing prompts"""

    model_config = ConfigDict(extra="forbid")

    prompts: list[PromptExampleResponse]
    count: int


class RenderPromptResponse(BaseModel):
    """Response schema for rendered prompt"""

    model_config = ConfigDict(extra="forbid")

    rendered_messages: list[PromptMessage]
    used_version: int
    missing_vars: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
