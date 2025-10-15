"""
SQLAlchemy model for app.runs table
"""
from sqlalchemy import Column, BigInteger, Integer, String, Text, Boolean, TIMESTAMP, Numeric, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = {"schema": "app"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    occurred_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))
    trace_id = Column(Text, nullable=False)
    request_id = Column(Text)
    session_id = Column(Text)
    user_id = Column(Text)
    
    provider_key = Column(Text, nullable=False)
    model_id = Column(Text, nullable=False)
    prompt_key = Column(Text)
    prompt_version = Column(Text)
    technique_key = Column(Text)
    
    params_json = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    variables_json = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    input_text = Column(Text)
    output_text = Column(Text)
    output_preview = Column(Text)  # Generated column: first 400 chars of output_text
    
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    reasoning_tokens = Column(Integer)  # Generated column: total_tokens - (prompt_tokens + completion_tokens)
    cost_usd = Column(Numeric(12, 6))
    
    latency_ms = Column(Integer)
    ttft_ms = Column(Integer)
    status = Column(Text, nullable=False)
    error_type = Column(Text)
    error_code = Column(Text)
    error_message = Column(Text)
    
    cached = Column(Boolean, nullable=False, server_default=text("false"))
    pricing_snapshot = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    metadata_json = Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb"))

    def __repr__(self):
        return f"<Run(id={self.id}, trace_id='{self.trace_id}', model='{self.model_id}', status='{self.status}')>"

