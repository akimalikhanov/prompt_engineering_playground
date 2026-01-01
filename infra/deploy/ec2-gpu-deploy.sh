#!/usr/bin/env bash
set -euo pipefail

log() { echo "[$(date -Is)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

INSTALL_ROOT="${INSTALL_ROOT:-/opt/vllm}"
AWS_REGION="${AWS_REGION:-${REGION:-eu-north-1}}"

# SSM path to HF_TOKEN parameter (single parameter, not a path prefix)
SSM_HF_TOKEN_PATH="${SSM_HF_TOKEN_PATH:-/pep/prod/HF_TOKEN}"

MODELS="${MODELS:-qwen}"

COMPOSE_FILE="${COMPOSE_FILE:-$INSTALL_ROOT/infra/deploy/vllm-docker-compose-prod.yml}"

usage() {
  cat <<'EOF'
Usage: ec2-gpu-deploy.sh [options]

Deploy vLLM models to GPU EC2 instance.
Pulls vLLM images from DockerHub (not ECR).

Options (or env vars with same name):
  --install-root PATH           (INSTALL_ROOT)      default: /opt/vllm
  --region AWS_REGION           (AWS_REGION)        default: eu-north-1
  --compose-file PATH           (COMPOSE_FILE)      default: $INSTALL_ROOT/infra/deploy/vllm-docker-compose-prod.yml
  --ssm-hf-token-path PATH      (SSM_HF_TOKEN_PATH) default: /pep/prod/HF_TOKEN
  --models MODELS               (MODELS)            default: qwen (comma-separated: qwen,llama)

Example:
  ./infra/deploy/ec2-gpu-deploy.sh --models qwen,llama
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-root) INSTALL_ROOT="$2"; shift 2 ;;
    --region) AWS_REGION="$2"; shift 2 ;;
    --compose-file) COMPOSE_FILE="$2"; shift 2 ;;
    --ssm-hf-token-path) SSM_HF_TOKEN_PATH="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done

need aws
need docker

if ! docker compose version >/dev/null 2>&1; then
  die "Docker Compose plugin not found. Install Docker Compose v2 (docker compose ...)."
fi

# Verify nvidia-smi is available (GPU instance check)
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "WARNING: nvidia-smi not found. This may not be a GPU instance."
else
  log "GPU check: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

log "1) Fetching HF_TOKEN from SSM"
log "   SSM path: $SSM_HF_TOKEN_PATH"
log "   Region:   $AWS_REGION"

HF_TOKEN="$(aws ssm get-parameter \
  --region "$AWS_REGION" \
  --name "$SSM_HF_TOKEN_PATH" \
  --with-decryption \
  --query 'Parameter.Value' \
  --output text)"

if [[ -z "$HF_TOKEN" ]]; then
  die "Failed to fetch HF_TOKEN from SSM path: $SSM_HF_TOKEN_PATH"
fi

export HF_TOKEN
log "   HF_TOKEN fetched successfully"

log "2) Checking files"
log "   Install root: $INSTALL_ROOT"
log "   Models:       $MODELS"

[[ -f "$COMPOSE_FILE" ]] || die "Compose file not found: $COMPOSE_FILE"

log "   Available vLLM configs:"
for cfg in "$INSTALL_ROOT"/config/vllm*.yaml; do
  [[ -f "$cfg" ]] && log "     - $(basename "$cfg")"
done

log "3) Parsing models to deploy"
# Parse comma-separated models into array
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
SERVICES=()
for model in "${MODEL_ARRAY[@]}"; do
  model="$(echo "$model" | tr -d '[:space:]')"
  config_file="$INSTALL_ROOT/config/vllm_config_${model}.yaml"
  
  if [[ -f "$config_file" ]]; then
    SERVICES+=("vllm-${model}")
    log "   -> Will deploy: vllm-${model} (config: vllm_config_${model}.yaml)"
  else
    log "   WARNING: Config file not found for model '$model' (expected: $config_file), skipping"
  fi
done

if [[ ${#SERVICES[@]} -eq 0 ]]; then
  die "No valid models specified. Use: qwen, llama"
fi

log "4) Stopping existing vLLM containers (if any)"
# Stop only vllm containers, not other containers that might be running
docker ps -a --filter "name=vllm-" --format "{{.Names}}" | xargs -r docker rm -f || true

log "5) Smart cleanup (keep vllm images, prune dangling)"
# Remove stopped containers (not running ones)
docker container prune -f || true
# Remove dangling images (untagged, not used)
docker image prune -f || true
# Remove unused networks
docker network prune -f || true
# Remove build cache
docker builder prune -f || true
# DO NOT prune volumes (keep hf-cache with downloaded models)

log "6) docker compose pull (from DockerHub)"
# Export INSTALL_ROOT as REPO_DIR for compose file volume paths
export REPO_DIR="$INSTALL_ROOT"

# Pull only the services we need
docker compose -f "$COMPOSE_FILE" pull "${SERVICES[@]}"

log "7) docker compose up (selected models only)"
docker compose -f "$COMPOSE_FILE" up -d --force-recreate "${SERVICES[@]}"

log "8) Waiting for containers to start"
sleep 5

log "9) Container status:"
docker ps --filter "name=vllm-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

log "10) Done - vLLM deployment complete"
log "   Models deployed: ${MODELS}"
