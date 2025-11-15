#!/bin/sh
set -eu

echo "[init] granting permissions to app schema..."

# This script can be run directly or via docker-compose
# If running directly, ensure APP_DB_USER and APP_DB are set
if [ -z "${APP_DB_USER:-}" ] || [ -z "${APP_DB:-}" ]; then
    echo "Error: APP_DB_USER and APP_DB must be set"
    exit 1
fi

psql -v ON_ERROR_STOP=1 --username "$APP_DB_USER" --dbname "$APP_DB" <<-'EOSQL'
-- Ensure schema ownership
ALTER SCHEMA app OWNER TO CURRENT_USER;

-- Grant schema usage
GRANT USAGE ON SCHEMA app TO CURRENT_USER;

-- Grant permissions on all existing tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA app TO CURRENT_USER;

-- Grant permissions on all existing sequences
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA app TO CURRENT_USER;

-- Grant permissions on all existing functions
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA app TO CURRENT_USER;

-- Grant permissions on all existing views
GRANT SELECT ON ALL TABLES IN SCHEMA app TO CURRENT_USER;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON TABLES TO CURRENT_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON SEQUENCES TO CURRENT_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON FUNCTIONS TO CURRENT_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT SELECT ON VIEWS TO CURRENT_USER;
EOSQL

echo "[init] permissions granted."

