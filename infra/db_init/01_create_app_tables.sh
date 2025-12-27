#!/bin/sh
set -eu

echo "[init] creating app schema and tables..."

psql -v ON_ERROR_STOP=1 --username "$APP_DB_USER" --dbname "$APP_DB" <<-'EOSQL'
-- Ensures UUID generator and schema
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS app;

-- ===============================
-- Prompt examples
-- ===============================
CREATE TABLE IF NOT EXISTS app.prompt_examples (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- internal key you can hardcode in code / seeds
    key                 TEXT NOT NULL,
    version             INT NOT NULL DEFAULT 1,

    -- basic metadata
    title               TEXT NOT NULL,
    description         TEXT,
    category            TEXT,                    -- e.g. 'qa', 'summarization', 'coding'
    technique           TEXT NOT NULL CHECK (
                            technique IN ('zero_shot', 'few_shot', 'prompt_chain')
                        ),
    tags                TEXT[] DEFAULT '{}',     -- optional

    -- core template (messages array in LLM format with Jinja variables)
    prompt_template     JSONB NOT NULL,          -- [{"role": "system", "content": "..."}, {"role": "user", "content": "{{variable}}"}]
    -- Messages array format with Jinja variables in content fields

    -- variable definitions (for rendering UI form & defaults)
    variables           JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- [
    --   {"name": "input_text", "type": "string", "default": "", "required": true},
    --   {"name": "tone", "type": "string", "default": "neutral", "required": false}
    -- ]

    -- few-shot defaults (optional; user can add more examples in UI at runtime)
    default_examples    JSONB,                   -- NULL if zero-shot
    -- e.g. [{"input": "Q1", "output": "A1"}, {"input": "Q2", "output": "A2"}]

    -- response format for structured output
    response_format     TEXT CHECK (
                            response_format IN ('json_object', 'json_schema')
                        ),
    -- NULL = no structured output, 'json_object' = basic JSON mode, 'json_schema' = JSON schema mode (preferred)

    -- JSON schema template (only used when response_format='json_schema')
    json_schema_template JSONB,  -- NULL if not using json_schema, JSON schema definition if using json_schema

    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    tool_config JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Each key can have multiple versions, but version numbers must be unique per key
    UNIQUE (key, version)
);

CREATE INDEX IF NOT EXISTS idx_prompt_examples_technique ON app.prompt_examples(technique);
CREATE INDEX IF NOT EXISTS idx_prompt_examples_category ON app.prompt_examples(category);
-- simple full-text-ish search (search in title and description only, prompt_template is JSONB now)
CREATE INDEX IF NOT EXISTS idx_prompt_examples_search
ON app.prompt_examples
USING GIN (to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(description,'')));

-- ===============================
-- Runs (analytics-oriented)
-- ===============================
CREATE TABLE IF NOT EXISTS app.runs (
  id                BIGSERIAL PRIMARY KEY,
  occurred_at       TIMESTAMPTZ NOT NULL DEFAULT now(),            -- when the run completed
  trace_id          TEXT NOT NULL,                                  -- correlate to logs/traces
  request_id        TEXT,                                           -- unique per call if available
  session_id        TEXT,                                           -- chat/session
  provider_key      TEXT NOT NULL,                                  -- 'openai','google','anthropic',...
  model_id          TEXT NOT NULL,                                  -- 'gpt-4o-mini', etc.

  params_json       JSONB NOT NULL DEFAULT '{}'::jsonb,             -- temperature, top_p, tools
  input_text        TEXT,                                           -- sanitized user input
  system_prompt     TEXT,                                           -- system instructions applied to the run
  context_prompt    TEXT,                                           -- supplemental context provided with the request
  output_text       TEXT,                                           -- sanitized final output
  output_preview    TEXT GENERATED ALWAYS AS
                     (left(coalesce(output_text,''), 400)) STORED,

  prompt_tokens     INTEGER,
  completion_tokens INTEGER,
  total_tokens      INTEGER,
  reasoning_tokens  INTEGER,
  cost_usd          NUMERIC(12,6),

  latency_ms        INTEGER,
  ttft_ms           INTEGER,
  is_stream         BOOLEAN GENERATED ALWAYS AS (ttft_ms IS NOT NULL AND tokens_per_second IS NOT NULL) STORED,
  tokens_per_second  NUMERIC(10,2),
  status            TEXT NOT NULL CHECK (status IN ('ok','error','rate_limited','timeout','cancelled')),
  error_type        TEXT,
  error_code        TEXT,
  error_message     TEXT,

  user_feedback     SMALLINT NOT NULL DEFAULT 0 CHECK (user_feedback IN (-1, 0, 1)),  -- -1 = negative, 0 = neutral/default, 1 = positive
  tool_call         JSONB,                                           -- JSON array or object of user tool names; NULL when unused

  cached            BOOLEAN NOT NULL DEFAULT FALSE
);

-- Indexes for BI
CREATE UNIQUE INDEX IF NOT EXISTS ux_runs_request_id
  ON app.runs (request_id) WHERE request_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_runs_time            ON app.runs (occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_runs_status_time     ON app.runs (status, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_runs_trace           ON app.runs (trace_id);
CREATE INDEX IF NOT EXISTS ix_runs_provider_model  ON app.runs (provider_key, model_id, occurred_at DESC);

-- Light JSONB indexes for common filters
CREATE INDEX IF NOT EXISTS gin_runs_params         ON app.runs USING GIN (params_json);

-- Grant permissions to the current user (APP_DB_USER)
-- Since we're running as APP_DB_USER, ensure schema ownership and permissions
ALTER SCHEMA app OWNER TO CURRENT_USER;
GRANT USAGE ON SCHEMA app TO CURRENT_USER;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA app TO CURRENT_USER;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA app TO CURRENT_USER;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA app TO CURRENT_USER;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON TABLES TO CURRENT_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON SEQUENCES TO CURRENT_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON FUNCTIONS TO CURRENT_USER;
EOSQL

echo "[init] app tables created."
