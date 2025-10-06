#!/bin/sh
set -eu

echo "[init] creating users and databases…"

# create users first
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
  DO \$\$
  BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${MLFLOW_DB_USER}') THEN
      CREATE USER ${MLFLOW_DB_USER} WITH PASSWORD '${MLFLOW_DB_PASSWORD}';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${APP_DB_USER}') THEN
      CREATE USER ${APP_DB_USER} WITH PASSWORD '${APP_DB_PASSWORD}';
    END IF;
  END
  \$\$;
EOSQL

# then create databases separately (outside DO)
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
  CREATE DATABASE ${MLFLOW_DB} OWNER ${MLFLOW_DB_USER};
  CREATE DATABASE ${APP_DB} OWNER ${APP_DB_USER};
EOSQL

echo "[init] done."
