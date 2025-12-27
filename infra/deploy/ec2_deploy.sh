#!/usr/bin/env bash
set -euo pipefail

# EC2 deployment script for prompt_engineering_playground
#
# IMPORTANT: This script assumes the repo is already cloned and checked out
# to the correct commit by the calling workflow (cd-ec2.yml).
#
# What it does:
# 1) Verify repo exists at expected commit (if --commit-sha provided)
# 2) Pull env vars from SSM Parameter Store into a local .env file
# 3) Login to any ECR registries referenced by the compose file (after env expansion)
# 4) Pull images from docker-compose-prod.yml
# 5) docker compose up -d --no-build --force-recreate --remove-orphans
# 6) docker image prune -f
# 7) done

log() { echo "[$(date -Is)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

# Host paths used by docker-compose-prod.yml default to /opt/pep/...
INSTALL_ROOT="${INSTALL_ROOT:-/opt/pep}"
REPO_DIR="${REPO_DIR:-$INSTALL_ROOT/prompt_engineering_playground}"

SSM_BASE_PATH="${SSM_BASE_PATH:-/pep/prod/env}"
AWS_REGION="${AWS_REGION:-${REGION:-eu-north-1}}"

# Optional: expected commit SHA for verification
COMMIT_SHA="${COMMIT_SHA:-}"
IMAGE_TAG="${IMAGE_TAG:-}"

# Compose & env file locations (can be overridden)
COMPOSE_FILE="${COMPOSE_FILE:-$REPO_DIR/infra/deploy/docker-compose-prod.yml}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"

usage() {
  cat <<'EOF'
Usage: ec2_deploy.sh [options]

This script deploys the application. It assumes the repo is already cloned
and checked out to the correct commit by the calling workflow.

Options (or env vars with same name):
  --install-root PATH       (INSTALL_ROOT)  default: /opt/pep
  --repo-dir PATH           (REPO_DIR)      default: $INSTALL_ROOT/prompt_engineering_playground
  --ssm-path PATH           (SSM_BASE_PATH) default: /pep/prod/env
  --region AWS_REGION       (AWS_REGION)    default: eu-north-1
  --compose-file PATH       (COMPOSE_FILE)  default: $REPO_DIR/infra/deploy/docker-compose-prod.yml
  --env-file PATH           (ENV_FILE)      default: $REPO_DIR/.env
  --commit-sha SHA          (COMMIT_SHA)    optional: verify repo is at this commit
  --image-tag TAG           (IMAGE_TAG)     optional: for logging purposes
  -h, --help                Show help

Ignored (for backward compatibility):
  --repo-url, --branch      These are handled by the workflow, not this script.

Example:
  AWS_REGION=eu-north-1 SSM_BASE_PATH=/pep/prod/env ./infra/deploy/ec2_deploy.sh --commit-sha abc123
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    # Ignored flags (repo management is done by workflow)
    --repo-url) shift 2 ;;
    --branch) shift 2 ;;
    # Active flags
    --install-root) INSTALL_ROOT="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --ssm-path) SSM_BASE_PATH="$2"; shift 2 ;;
    --region) AWS_REGION="$2"; shift 2 ;;
    --compose-file) COMPOSE_FILE="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --commit-sha) COMMIT_SHA="$2"; shift 2 ;;
    --image-tag) IMAGE_TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done

need git
need aws
need docker

if ! docker compose version >/dev/null 2>&1; then
  die "Docker Compose plugin not found. Install Docker Compose v2 (docker compose ...)."
fi

log "1) Verifying repo state"
[[ -d "$REPO_DIR/.git" ]] || die "Repo not found at $REPO_DIR - workflow should have cloned it"

ACTUAL_SHA="$(git -C "$REPO_DIR" rev-parse HEAD)"
log "   Repo dir:    $REPO_DIR"
log "   Current SHA: $ACTUAL_SHA"
if [[ -n "$IMAGE_TAG" ]]; then
  log "   Image tag:   $IMAGE_TAG"
fi

if [[ -n "$COMMIT_SHA" ]]; then
  # Compare short or full SHA
  if [[ "$ACTUAL_SHA" != "$COMMIT_SHA"* && "$COMMIT_SHA" != "$ACTUAL_SHA"* ]]; then
    die "Commit mismatch! Expected: $COMMIT_SHA, Actual: $ACTUAL_SHA"
  fi
  log "   Commit verified: $COMMIT_SHA"
fi

# Default IMAGE_TAG to commit SHA if not provided
if [[ -z "$IMAGE_TAG" ]]; then
  IMAGE_TAG="$ACTUAL_SHA"
  log "   IMAGE_TAG defaulted to: $IMAGE_TAG"
fi
export IMAGE_TAG

[[ -f "$COMPOSE_FILE" ]] || die "Compose file not found: $COMPOSE_FILE"

log "2) Fetching SSM params -> .env"
mkdir -p "$INSTALL_ROOT"
log "   SSM path: $SSM_BASE_PATH"
log "   Region:   $AWS_REGION"

TMP_JSON="$(mktemp)"
trap 'rm -f "$TMP_JSON"' EXIT

aws ssm get-parameters-by-path \
  --region "$AWS_REGION" \
  --with-decryption \
  --recursive \
  --path "$SSM_BASE_PATH" \
  --output json >"$TMP_JSON"

mkdir -p "$(dirname "$ENV_FILE")"

if command -v python3 >/dev/null 2>&1; then
  SSM_BASE_PATH="$SSM_BASE_PATH" ENV_FILE="$ENV_FILE" python3 - "$TMP_JSON" <<'PY'
import json, os, sys

json_path = sys.argv[1]
with open(json_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

prefix = os.environ["SSM_BASE_PATH"].rstrip("/") + "/"
env_path = os.environ["ENV_FILE"]

params = payload.get("Parameters", [])
items = []
for p in params:
    name = p.get("Name", "")
    value = p.get("Value", "")
    if not name.startswith(prefix):
        continue
    key = name[len(prefix):]
    # Docker/Compose .env parser is line-based; represent newlines safely.
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    value = value.replace("\\", "\\\\").replace('"', '\\"')
    items.append((key, value))

items.sort(key=lambda kv: kv[0])

with open(env_path, "w", encoding="utf-8") as out:
    out.write("# Generated by infra/deploy/ec2_deploy.sh\n")
    out.write(f"# Source: SSM {prefix[:-1]}\n")
    out.write("\n")
    for k, v in items:
        out.write(f'{k}="{v}"\n')
PY
elif command -v jq >/dev/null 2>&1; then
  # Best-effort fallback; values are quoted with JSON escaping removed via jq.
  prefix="${SSM_BASE_PATH%/}/"
  {
    echo "# Generated by infra/deploy/ec2_deploy.sh"
    echo "# Source: SSM ${prefix%/}"
    echo
    jq -r --arg p "$prefix" '
      .Parameters
      | map(select(.Name | startswith($p)))
      | sort_by(.Name)
      | .[]
      | (.Name | sub("^" + $p; "")) + "=" + ( .Value | gsub("\r\n|\r|\n"; "\\n") | @json )
    ' "$TMP_JSON" | sed -E 's/=(.+)$/=\1/'  # already quoted by @json
  } >"$ENV_FILE"
else
  die "Need python3 (preferred) or jq to convert SSM JSON into a .env file"
fi

chmod 600 "$ENV_FILE" || true
log "   Wrote: $ENV_FILE"

log "3) Logging into ECR registries referenced by compose (after env expansion)"
TMP_COMPOSE_CFG="$(mktemp)"
trap 'rm -f "$TMP_JSON" "$TMP_COMPOSE_CFG"' EXIT

docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" config >"$TMP_COMPOSE_CFG"

mapfile -t ECR_REGISTRIES < <(
  grep -oE '[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com' "$TMP_COMPOSE_CFG" \
    | sort -u
)

if [[ ${#ECR_REGISTRIES[@]} -eq 0 ]]; then
  log "   No private ECR registries found in expanded compose config (skipping ECR login)."
else
  for reg in "${ECR_REGISTRIES[@]}"; do
    region="$(echo "$reg" | cut -d. -f4)"
    log "   -> docker login $reg (region: $region)"
    aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$reg" >/dev/null
  done
fi

log "4) Stopping and removing old containers"
# Stop all containers from this compose project
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down --remove-orphans || true

log "5) Cleaning up disk space"
# Remove all stopped containers
docker container prune -f || true
# Remove all unused images (not just dangling)
docker image prune -a -f || true
# Remove unused volumes (careful: this removes data volumes too)
# docker volume prune -f || true
# Remove build cache
docker builder prune -f || true

log "6) docker compose pull"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" pull

log "7) docker compose up"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d --no-build --force-recreate --remove-orphans

log "8) Final cleanup (dangling images)"
docker image prune -f || true

log "9) Done"

