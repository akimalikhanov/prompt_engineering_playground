#!/usr/bin/env bash
set -euo pipefail

# --- Config (paths are relative to repo root) ---
COMPOSE_FILE="infra/docker-compose.yml"
ENV_FILE=".env"
POSTGRES_DIR="storage/postgres"
MINIO_DIR="storage/minio"
CLEAN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --clean)
      CLEAN=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--clean]"
      exit 1
      ;;
  esac
done

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

# --- Clean storage if requested ---
if [ "$CLEAN" = true ]; then
  log "Cleaning storage directories (--clean flag provided)"
  maybe_sudo rm -rf "$POSTGRES_DIR"
  # MinIO files may be owned by container (root), so use sudo or Docker
  if have_sudo && sudo -n true 2>/dev/null; then
    maybe_sudo rm -rf "$MINIO_DIR"
  else
    log "Using Docker to clean MinIO directory (sudo not available)"
    docker run --rm -v "$(pwd)/$MINIO_DIR:/data" alpine sh -c "rm -rf /data/* /data/.[!.]* /data/..?*" 2>/dev/null || true
    rm -rf "$MINIO_DIR" 2>/dev/null || true
  fi
fi

# --- Folders & permissions ---
log "Preparing storage directories"
mkdir -p "$POSTGRES_DIR" "$MINIO_DIR"
# Postgres inside container runs as uid:gid 999:999 on alpine image
# Use Docker to set permissions if sudo is not available non-interactively
if have_sudo && sudo -n true 2>/dev/null; then
  maybe_sudo chown -R 999:999 "$POSTGRES_DIR"
  maybe_sudo chmod 700 "$POSTGRES_DIR"
else
  log "Using Docker to set postgres directory permissions (sudo not available)"
  docker run --rm -v "$(pwd)/$POSTGRES_DIR:/data" alpine sh -c "chown -R 999:999 /data && chmod 700 /data"
fi
chmod -R u+rwX "$MINIO_DIR" 2>/dev/null || true

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
  if [ "$pg_state" = "healthy" ]; then
    log "✅ Postgres is healthy"
    break
  fi
  sleep 2
done
if [ "$pg_state" != "healthy" ]; then
  echo "Postgres did not become healthy in time. Last state: $pg_state"
  exit 1
fi

# --- Wait for MinIO ready (HTTP readiness endpoint on host port) ---
MINIO_API_PORT="$(grep -E '^MINIO_API_PORT=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || echo 9000)"
log "Waiting for MinIO (S3 API) readiness on http://localhost:${MINIO_API_PORT}/minio/health/ready"
for i in {1..60}; do
  if curl -fsS http://localhost:${MINIO_API_PORT}/minio/health/ready >/dev/null 2>&1; then
    log "✅ MinIO is ready"
    break
  fi
  sleep 2
done
if ! curl -fsS http://localhost:${MINIO_API_PORT}/minio/health/ready >/dev/null 2>&1; then
  echo "MinIO did not become ready in time."
  exit 1
fi

# --- Run MinIO bootstrap (bucket/user/policy) ---
log "Bootstrapping MinIO (minio-init)"
"${DC[@]}" run --rm minio-init

# --- Start MLflow server ---
log "Starting MLflow server"
"${DC[@]}" up -d mlflow

# --- Start OpenTelemetry Collector ---
log "Starting OpenTelemetry Collector"
"${DC[@]}" up -d otel-collector

# --- Wait for Collector health endpoint ---
OTEL_HEALTH_PORT="$(grep -E '^OTEL_HEALTH_PORT=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || echo 13133)"
log "Waiting for OpenTelemetry Collector health on http://localhost:${OTEL_HEALTH_PORT}/"
for i in {1..60}; do
  if curl -fsS http://localhost:${OTEL_HEALTH_PORT}/ >/dev/null 2>&1; then
    log "✅ OTEL Collector is healthy"
    break
  fi
  sleep 2
done
if ! curl -fsS http://localhost:${OTEL_HEALTH_PORT}/ >/dev/null 2>&1; then
  echo "OpenTelemetry Collector did not become healthy in time."
  exit 1
fi

# --- Start Grafana ---
log "Starting Grafana"
"${DC[@]}" up -d grafana

# --- Show status & useful URLs ---
log "Compose status"
"${DC[@]}" ps

# Pull ports from .env for display (defaults are safe fallbacks)
POSTGRES_PORT="$(grep -E '^POSTGRES_PORT=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || echo 5433)"
MINIO_CONSOLE_PORT="$(grep -E '^MINIO_CONSOLE_PORT=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || echo 9001)"
MLFLOW_PORT="$(grep -E '^MLFLOW_PORT=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || echo 5000)"
GRAFANA_PORT="$(grep -E '^GRAFANA_PORT=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || echo 3000)"
TEMPO_HTTP_PORT="$(grep -E '^TEMPO_HTTP_PORT=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || echo 3200)"
OTEL_HTTP_PORT="$(grep -E '^OTEL_HTTP_PORT=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || echo 4318)"

cat <<EOF

✅ All set!

Services:
- Postgres:          localhost:${POSTGRES_PORT} (container port 5432)
- MinIO S3 API:      http://localhost:${MINIO_API_PORT}
- MinIO Console:     http://localhost:${MINIO_CONSOLE_PORT} (minioadmin/minioadmin)
- MLflow Tracking:   http://localhost:${MLFLOW_PORT}
- Grafana:           http://localhost:${GRAFANA_PORT} (admin/admin)
- Tempo:             http://localhost:${TEMPO_HTTP_PORT}
- OTEL Collector:    http://localhost:${OTEL_HTTP_PORT} (OTLP/HTTP)
EOF
