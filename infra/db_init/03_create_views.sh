#!/bin/sh
set -eu

echo "[init] creating views..."

psql -v ON_ERROR_STOP=1 --username "$APP_DB_USER" --dbname "$APP_DB" <<-'EOSQL'
-- View: Latest enabled prompts (one per technique_key + title)
CREATE OR REPLACE VIEW app.v_prompt_examples_latest AS
SELECT 
  pe.*
FROM app.prompt_examples pe
INNER JOIN (
  SELECT 
    technique_key,
    title,
    MAX(version) as max_version
  FROM app.prompt_examples
  WHERE status = 'active' AND is_enabled = true
  GROUP BY technique_key, title
) latest
ON pe.technique_key = latest.technique_key
   AND pe.title = latest.title
   AND pe.version = latest.max_version;

COMMENT ON VIEW app.v_prompt_examples_latest IS 
'Returns only the latest active and enabled version of each prompt (grouped by technique_key + title)';

EOSQL

echo "[init] views created."

