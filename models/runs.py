"""
SQLAlchemy model for app.runs table
"""

from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    Boolean,
    Column,
    Computed,
    Integer,
    Numeric,
    SmallInteger,
    Text,
    text,
)
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

    provider_key = Column(Text, nullable=False)
    model_id = Column(Text, nullable=False)

    params_json = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    input_text = Column(Text)
    system_prompt = Column(Text)
    context_prompt = Column(Text)
    output_text = Column(Text)
    output_preview = Column(
        Text, Computed("left(coalesce(output_text,''), 400)")
    )  # Generated column: first 400 chars of output_text

    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    reasoning_tokens = Column(Integer)
    cost_usd = Column(Numeric(12, 6))

    latency_ms = Column(Integer)
    ttft_ms = Column(Integer)
    is_stream = Column(
        Boolean, Computed("ttft_ms IS NOT NULL AND tokens_per_second IS NOT NULL")
    )  # Generated column: TRUE if streaming (both ttft_ms and tokens_per_second are not null)
    tokens_per_second = Column(Numeric(10, 2))
    status = Column(Text, nullable=False)
    error_type = Column(Text)
    error_code = Column(Text)
    error_message = Column(Text)

    cached = Column(Boolean, nullable=False, server_default=text("false"))

    # User feedback and tool calls (added later)
    user_feedback = Column(SmallInteger, nullable=False, server_default=text("0"))
    tool_call = Column(JSONB)

    def __repr__(self):
        return f"<Run(id={self.id}, trace_id='{self.trace_id}', model='{self.model_id}', status='{self.status}')>"
