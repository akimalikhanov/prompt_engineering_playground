#!/usr/bin/env bash
set -euo pipefail

# --- Config (paths are relative to repo root) ---
COMPOSE_FILE="infra/docker-compose.yml"
ENV_FILE=".env"
POSTGRES_DIR="storage/postgres"
MINIO_DIR="storage/minio"

# --- Helpers ---
need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1"; exit 1; }; }
have_sudo() { command -v sudo >/dev/null 2>&1; }
maybe_sudo() { if have_sudo; then sudo "$@"; else "$@"; fi; }
log() { printf "\n\033[1;36m▶ %s\033[0m\n" "$*"; }

DC=(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE")

# --- Pre-flight checks ---
need_cmd docker
need_cmd bash
[ -f "$COMPOSE_FILE" ] || { echo "Compose file not found: $COMPOSE_FILE"; exit 1; }
[ -f "$ENV_FILE" ]     || { echo ".env file not found: $ENV_FILE"; exit 1; }

# --- Folders & permissions ---
log "Preparing storage directories"
mkdir -p "$POSTGRES_DIR" "$MINIO_DIR"
# Postgres inside container runs as uid:gid 999:999 on alpine image
maybe_sudo chown -R 999:999 "$POSTGRES_DIR"
maybe_sudo chmod 700 "$POSTGRES_DIR"
chmod -R u+rwX "$MINIO_DIR"

# --- Fresh (re)start baseline (down doesn't touch volumes we just set up) ---
log "Stopping any existing stack (if running)"
"${DC[@]}" down --remove-orphans || true

# --- Build MLflow image first (only mlflow has 'build') ---
log "Building MLflow image (if needed)"
"${DC[@]}" build mlflow

# --- Start dependencies: Postgres & MinIO ---
log "Starting postgres"
"${DC[@]}" up -d postgres

log "Starting minio"
"${DC[@]}" up -d minio

# --- Wait for Postgres healthy (uses compose healthcheck) ---
log "Waiting for postgres to become healthy"
pg_state=""
for i in {1..60}; do
  pg_state="$(docker inspect -f '{{.State.Health.Status}}' mlflow-postgres 2>/dev/null || echo 'unknown')"
  [ "$pg_state" = "healthy" ] && break
  sleep 2
done
if [ "$pg_state" != "healthy" ]; then
  echo "Postgres did not become healthy in time. Last state: $pg_state"
  exit 1
fi

# --- Wait for MinIO ready (HTTP readiness endpoint on host port 9000) ---
log "Waiting for MinIO (S3 API) readiness on http://localhost:9000/minio/health/ready"
for i in {1..60}; do
  if curl -fsS http://localhost:9000/minio/health/ready >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if ! curl -fsS http://localhost:9000/minio/health/ready >/dev/null 2>&1; then
  echo "MinIO did not become ready in time."
  exit 1
fi

# --- Run MinIO bootstrap (bucket/user/policy) ---
log "Bootstrapping MinIO (minio-init)"
"${DC[@]}" run --rm minio-init

# --- Start MLflow server ---
log "Starting MLflow server"
"${DC[@]}" up -d mlflow

# --- Show status & useful URLs ---
log "Compose status"
"${DC[@]}" ps

# Pull ports from .env for display (defaults are safe fallbacks)
MLFLOW_PORT="$(grep -E '^MLFLOW_PORT=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || echo 5000)"

cat <<EOF

✅ All set!

Services:
- Postgres:          localhost:5433 (container port 5432)
- MinIO S3 API:      http://localhost:9000
- MinIO Console:     http://localhost:9001
- MLflow Tracking:   http://localhost:${MLFLOW_PORT}
EOF
