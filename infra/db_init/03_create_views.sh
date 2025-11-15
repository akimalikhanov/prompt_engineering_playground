#!/bin/sh
set -eu

echo "[init] creating views..."

psql -v ON_ERROR_STOP=1 --username "$APP_DB_USER" --dbname "$APP_DB" <<-'EOSQL'
-- View: Latest enabled prompts (one per key)
CREATE OR REPLACE VIEW app.v_prompt_examples_latest AS
SELECT 
  pe.*
FROM app.prompt_examples pe
INNER JOIN (
  SELECT 
    key,
    MAX(version) as max_version
  FROM app.prompt_examples
  WHERE is_active = true
  GROUP BY key
) latest
ON pe.key = latest.key
   AND pe.version = latest.max_version;

COMMENT ON VIEW app.v_prompt_examples_latest IS 
'Returns only the latest active version of each prompt (grouped by key)';

-- Grant permissions on views
GRANT SELECT ON app.v_prompt_examples_latest TO CURRENT_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT SELECT ON VIEWS TO CURRENT_USER;
EOSQL

echo "[init] views created."

