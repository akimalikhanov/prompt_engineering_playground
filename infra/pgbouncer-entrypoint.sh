#!/usr/bin/env bash
set -e

echo "Generating PgBouncer config from environment..."

# Generate userlist.txt
cat >/etc/pgbouncer/userlist.txt <<EOF
"${APP_DB_USER}" "${APP_DB_PASSWORD}"
"${MLFLOW_DB_USER}" "${MLFLOW_DB_PASSWORD}"
"${POSTGRES_USER}" "${POSTGRES_PASSWORD}"
EOF

# Generate pgbouncer.ini
cat >/etc/pgbouncer/pgbouncer.ini <<EOF
[databases]
app     = host=postgres port=5432 dbname=${APP_DB}    user=${APP_DB_USER}    password=${APP_DB_PASSWORD}
mlflow  = host=postgres port=5432 dbname=${MLFLOW_DB} user=${MLFLOW_DB_USER} password=${MLFLOW_DB_PASSWORD}
postgres = host=postgres port=5432 dbname=${POSTGRES_DB} user=${POSTGRES_USER} password=${POSTGRES_PASSWORD}

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432

auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

pool_mode = session
default_pool_size = 20
min_pool_size = 0
max_client_conn = 200

ignore_startup_parameters = extra_float_digits
server_reset_query = DISCARD ALL

log_connections = 1
log_disconnections = 1
log_pooler_errors = 1
EOF

echo "PgBouncer config generated."
exec pgbouncer /etc/pgbouncer/pgbouncer.ini
