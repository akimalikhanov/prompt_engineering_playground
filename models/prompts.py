"""
SQLAlchemy models for app.prompt_techniques and app.prompt_examples tables
"""
from sqlalchemy import Column, String, Text, Integer, Boolean, TIMESTAMP, text, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
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
        UniqueConstraint('technique_key', 'title', 'version', name='prompt_examples_technique_key_title_version_key'),
        Index('ux_prompt_examples_active', 'technique_key', 'title', 
              postgresql_where=text("status = 'active'")),
        Index('idx_prompt_examples_latest', 'technique_key', 'title', 'version'),
        {"schema": "app"}
    )

    example_id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    technique_key = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    version = Column(Integer, nullable=False, server_default=text("1"))
    status = Column(Text, nullable=False, server_default=text("'active'"))
    language = Column(Text, nullable=False, server_default=text("'en'"))
    messages = Column(JSONB, nullable=False)
    variables = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    model_hint = Column(Text)
    is_enabled = Column(Boolean, nullable=False, server_default=text("true"))
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))

    def __repr__(self):
        return f"<PromptExample(technique_key='{self.technique_key}', title='{self.title}', version={self.version})>"

