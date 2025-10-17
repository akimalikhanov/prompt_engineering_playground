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
    
    technique_key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    language: Optional[str] = "en"
    messages: List[PromptMessage] = Field(min_items=1)
    variables: Optional[List[PromptVariable]] = []
    model_hint: Optional[str] = None


class CreateVersionRequest(BaseModel):
    """Request to create a new version of an existing prompt"""
    model_config = ConfigDict(extra="forbid")
    
    messages: Optional[List[PromptMessage]] = None
    variables: Optional[List[PromptVariable]] = None
    model_hint: Optional[str] = None
    language: Optional[str] = None


class RenderPromptRequest(BaseModel):
    """Request to render a prompt with variables"""
    model_config = ConfigDict(extra="forbid")
    
    variables: Dict[str, Any] = Field(default_factory=dict)


class PatchPromptRequest(BaseModel):
    """Request to update prompt status/enabled flag"""
    model_config = ConfigDict(extra="forbid")
    
    is_enabled: Optional[bool] = None
    status: Optional[Literal["active", "archived", "draft"]] = None


# ============================================
# Response schemas
# ============================================
class PromptExampleResponse(BaseModel):
    """Response schema for a prompt example"""
    model_config = ConfigDict(from_attributes=True)
    
    example_id: UUID
    technique_key: str
    title: str
    version: int
    status: str
    language: str
    messages: List[Dict[str, str]]  # Keep as dict for flexibility
    variables: List[Dict[str, Any]]  # Keep as dict for flexibility
    model_hint: Optional[str] = None
    is_enabled: bool
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

