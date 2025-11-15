"""
SQLAlchemy models for app.prompt_techniques and app.prompt_examples tables
"""
from sqlalchemy import Column, String, Text, Integer, Boolean, TIMESTAMP, text, UniqueConstraint, Index, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class PromptTechnique(Base):
    __tablename__ = "prompt_techniques"
    __table_args__ = {"schema": "app"}

    technique_id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    key = Column(Text, unique=True, nullable=False)
    name = Column(Text, nullable=False)
    family = Column(Text)
    url = Column(Text)
    short_desc = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))

    def __repr__(self):
        return f"<PromptTechnique(key='{self.key}', name='{self.name}')>"


class PromptExample(Base):
    __tablename__ = "prompt_examples"
    __table_args__ = (
        UniqueConstraint('key', 'version', name='prompt_examples_key_version_key'),
        CheckConstraint("technique IN ('zero_shot', 'few_shot', 'prompt_chain')", name='prompt_examples_technique_check'),
        CheckConstraint("response_format IN ('json_object', 'json_schema')", name='prompt_examples_response_format_check'),
        {"schema": "app"}
    )

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    key = Column(Text, nullable=False)
    version = Column(Integer, nullable=False, server_default=text("1"))
    title = Column(Text, nullable=False)
    description = Column(Text)
    category = Column(Text)
    technique = Column(Text, nullable=False)
    tags = Column(ARRAY(Text), server_default=text("'{}'"))
    prompt_template = Column(JSONB, nullable=False)  # Messages array: [{"role": "system", "content": "..."}, ...]
    variables = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    default_examples = Column(JSONB)
    response_format = Column(Text)  # NULL = no structured output, 'json_object' or 'json_schema'
    json_schema_template = Column(JSONB)  # JSON schema definition (only used when response_format='json_schema')
    tool_config = Column(JSONB)
    is_active = Column(Boolean, nullable=False, server_default=text("true"))
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))

    def __repr__(self):
        return f"<PromptExample(key='{self.key}', title='{self.title}', version={self.version})>"

