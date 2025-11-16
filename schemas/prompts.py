"""
Pydantic schemas for prompt management endpoints
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from uuid import UUID


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
    desc: Optional[str] = None


# ============================================
# Request schemas
# ============================================
class CreatePromptRequest(BaseModel):
    """Request to create a new prompt (v1)"""
    model_config = ConfigDict(extra="forbid")
    
    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    technique: str = Field(min_length=1)
    prompt_template: List[PromptMessage] = Field(min_items=1)
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    variables: Optional[List[PromptVariable]] = []
    default_examples: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[str] = None
    json_schema_template: Optional[Dict[str, Any]] = None
    tool_config: Optional[Dict[str, Any]] = None


class CreateVersionRequest(BaseModel):
    """Request to create a new version of an existing prompt"""
    model_config = ConfigDict(extra="forbid")
    
    prompt_template: Optional[List[PromptMessage]] = None
    variables: Optional[List[PromptVariable]] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    default_examples: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[str] = None
    json_schema_template: Optional[Dict[str, Any]] = None
    tool_config: Optional[Dict[str, Any]] = None


class RenderPromptRequest(BaseModel):
    """Request to render a prompt with variables"""
    model_config = ConfigDict(extra="forbid")
    
    variables: Dict[str, Any] = Field(default_factory=dict)


class PatchPromptRequest(BaseModel):
    """Request to update prompt status/enabled flag"""
    model_config = ConfigDict(extra="forbid")
    
    is_active: Optional[bool] = None


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
    prompt_template: List[Dict[str, str]]  # Messages array: [{"role": "system", "content": "..."}, ...]
    variables: List[Dict[str, Any]]  # Variable definitions
    is_active: bool
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    default_examples: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[str] = None
    json_schema_template: Optional[Dict[str, Any]] = None
    tool_config: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


class ListPromptsResponse(BaseModel):
    """Response schema for listing prompts"""
    model_config = ConfigDict(extra="forbid")
    
    prompts: List[PromptExampleResponse]
    count: int


class RenderPromptResponse(BaseModel):
    """Response schema for rendered prompt"""
    model_config = ConfigDict(extra="forbid")
    
    rendered_messages: List[PromptMessage]
    used_version: int
    missing_vars: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

