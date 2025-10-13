#!/bin/sh
set -eu

echo "[init] creating app schema and tables..."

psql -v ON_ERROR_STOP=1 --username "$APP_DB_USER" --dbname "$APP_DB" <<-'EOSQL'
-- Ensures UUID generator and schema
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS app;

-- ===============================
-- Techniques registry
-- ===============================
CREATE TABLE IF NOT EXISTS app.prompt_techniques (
  technique_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  key            TEXT UNIQUE NOT NULL,         -- e.g., 'cot','react','few-shot'
  name           TEXT NOT NULL,                -- 'Chain-of-Thought'
  family         TEXT,                         -- 'reasoning','instruction',...
  url            TEXT,                         -- reference link
  short_desc     TEXT NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ===============================
-- Prompt examples (multi-message)
-- ===============================
CREATE TABLE IF NOT EXISTS app.prompt_examples (
  example_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  technique_key  TEXT NOT NULL REFERENCES app.prompt_techniques(key) ON DELETE CASCADE,
  title          TEXT NOT NULL,                -- short label
  language       TEXT NOT NULL DEFAULT 'en',
  messages       JSONB NOT NULL,               -- [{role:'system|user|assistant|tool', content:'...'}, ...]
  variables      JSONB NOT NULL DEFAULT '[]',  -- [{"name":"text","type":"string","required":true,"desc":"..."}]
  model_hint     TEXT,
  is_enabled     BOOLEAN NOT NULL DEFAULT TRUE,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (technique_key, title)
);

CREATE INDEX IF NOT EXISTS idx_prompt_examples_technique ON app.prompt_examples(technique_key);

-- ===============================
-- Runs (analytics-oriented)
-- ===============================
CREATE TABLE IF NOT EXISTS app.runs (
  id                BIGSERIAL PRIMARY KEY,
  occurred_at       TIMESTAMPTZ NOT NULL DEFAULT now(),            -- when the run completed
  trace_id          TEXT NOT NULL,                                  -- correlate to logs/traces
  request_id        TEXT,                                           -- unique per call if available
  session_id        TEXT,                                           -- chat/session
  user_id           TEXT,                                           -- who initiated (optional)
  provider_key      TEXT NOT NULL,                                  -- 'openai','google','anthropic',...
  model_id          TEXT NOT NULL,                                  -- 'gpt-4o-mini', etc.
  prompt_key        TEXT,                                           -- e.g., 'system/coding'
  prompt_version    TEXT,                                           -- registry version/tag
  technique_key     TEXT,                                           -- 'cot','react','few-shot', etc.

  params_json       JSONB NOT NULL DEFAULT '{}'::jsonb,             -- temperature, top_p, tools
  variables_json    JSONB NOT NULL DEFAULT '[]'::jsonb,             -- templated vars used
  input_text        TEXT,                                           -- sanitized user input
  output_text       TEXT,                                           -- sanitized final output
  output_preview    TEXT GENERATED ALWAYS AS
                     (left(coalesce(output_text,''), 400)) STORED,

  prompt_tokens     INTEGER,
  completion_tokens INTEGER,
  total_tokens      INTEGER,
  cost_usd          NUMERIC(12,6),

  latency_ms        INTEGER,
  ttft_ms           INTEGER,
  status            TEXT NOT NULL CHECK (status IN ('ok','error','rate_limited','timeout','cancelled')),
  error_type        TEXT,
  error_code        TEXT,
  error_message     TEXT,

  cached            BOOLEAN NOT NULL DEFAULT FALSE,
  pricing_snapshot  JSONB NOT NULL DEFAULT '{}'::jsonb,             -- {"input_per_1k":0.15,...}
  metadata          JSONB NOT NULL DEFAULT '{}'::jsonb              -- tags/ab-test, biz labels
);

-- Indexes for BI
CREATE UNIQUE INDEX IF NOT EXISTS ux_runs_request_id
  ON app.runs (request_id) WHERE request_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_runs_time            ON app.runs (occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_runs_model_time      ON app.runs (model_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_runs_status_time     ON app.runs (status, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_runs_trace           ON app.runs (trace_id);
CREATE INDEX IF NOT EXISTS ix_runs_provider_model  ON app.runs (provider_key, model_id);

-- Light JSONB indexes for common filters
CREATE INDEX IF NOT EXISTS gin_runs_params         ON app.runs USING GIN (params_json);
CREATE INDEX IF NOT EXISTS gin_runs_metadata       ON app.runs USING GIN (metadata);
EOSQL

echo "[init] app tables created."

