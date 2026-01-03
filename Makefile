.PHONY: help up up-recreate down down-v restart ps logs build pull prune reset reset-volumes reset-postgres reset-minio reset-both vllm-up vllm-down vllm-logs

# --------
# Config
# --------
ENV_FILE ?= .env
COMPOSE_FILE ?= infra/docker-compose.yml
VLLM_COMPOSE_FILE ?= infra/vllm-docker-compose.yml

DC = docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE)
VLLM_DC = docker compose --env-file $(ENV_FILE) -f $(VLLM_COMPOSE_FILE)

help:
	@echo ""
	@echo "Prompt Engineering Playground"
	@echo ""
	@echo "Common commands:"
	@echo "  make up            Start main stack (api/ui/mlflow/postgres/minio/observability)"
	@echo "  make up-recreate   Start main stack with --force-recreate --remove-orphans"
	@echo "  make down          Stop main stack"
	@echo "  make down-v        Stop main stack and remove named volumes (-v)"
	@echo "  make logs          Tail logs from main stack"
	@echo "  make ps            Show container status"
	@echo "  make prune         Prune stopped containers (docker container prune)"
	@echo "  make reset         Stop stack and remove local data directories (DANGEROUS)"
	@echo "                   Examples: make reset WHAT=postgres|minio|both"
	@echo "                             make reset WHAT=both VOLUMES=1 PRUNE=1"
	@echo "                   Shortcuts: make reset-postgres | reset-minio | reset-both"
	@echo "                             make reset-volumes"
	@echo ""
	@echo "Optional:"
	@echo "  make vllm-up        Start local vLLM stack (GPU host needed)"
	@echo "  make vllm-down      Stop local vLLM stack"
	@echo "  make vllm-logs      Tail logs from local vLLM stack"
	@echo ""
	@echo "Notes:"
	@echo "  - Ensure $(ENV_FILE) exists (usually copy .env.example -> .env)."
	@echo "  - Override files like: make up ENV_FILE=.env.local"
	@echo ""

up:
	@$(DC) up -d --build

up-recreate:
	@$(DC) up -d --build --force-recreate --remove-orphans

down:
	@$(DC) down --remove-orphans

down-v:
	@$(DC) down -v --remove-orphans

restart: down up

ps:
	@$(DC) ps

logs:
	@$(DC) logs -f --tail=200

build:
	@$(DC) build

pull:
	@$(DC) pull

prune:
	@docker container prune -f

# Removes local data directories used by dev compose file (storage/*).
# This is intentionally separate from docker volume cleanup so it's explicit.
# Usage:
#   make reset WHAT=postgres|minio|both   (default: both)
#   make reset WHAT=both VOLUMES=1        (also remove compose volumes)
#   make reset WHAT=both PRUNE=1          (also prune stopped containers)
reset:
	@WHAT="$(WHAT)" bash -euo pipefail -c '\
		volumes="$${VOLUMES:-0}"; \
		if [ "$$volumes" = "1" ]; then \
			$(MAKE) down-v; \
		else \
			$(MAKE) down; \
		fi; \
		what="$${WHAT:-both}"; \
		case "$$what" in \
			postgres) dirs=(storage/postgres) ;; \
			minio)    dirs=(storage/minio) ;; \
			both)     dirs=(storage/postgres storage/minio) ;; \
			*) echo "Invalid WHAT='\''$$what'\'' (expected postgres|minio|both)"; exit 2 ;; \
		esac; \
		echo "Removing local data directories: $${dirs[*]}"; \
		rm -rf "$${dirs[@]}" 2>/dev/null || true; \
		remaining=(); \
		for d in "$${dirs[@]}"; do [ -e "$$d" ] && remaining+=("$$d"); done; \
		if [ "$${#remaining[@]}" -gt 0 ]; then \
			echo "Some paths could not be removed (likely root-owned): $${remaining[*]}"; \
			echo "Retrying via Docker..."; \
			rm_args=(); \
			for d in "$${remaining[@]}"; do rm_args+=("/storage/$${d#storage/}"); done; \
			docker run --rm -v "$$PWD/storage:/storage" alpine rm -rf "$${rm_args[@]}"; \
		fi; \
		prune="$${PRUNE:-0}"; \
		if [ "$$prune" = "1" ]; then \
			$(MAKE) prune; \
		fi \
	'

reset-volumes:
	@$(MAKE) reset VOLUMES=1

reset-postgres:
	@$(MAKE) reset WHAT=postgres

reset-minio:
	@$(MAKE) reset WHAT=minio

reset-both:
	@$(MAKE) reset WHAT=both

vllm-up:
	@$(VLLM_DC) up -d

vllm-down:
	@$(VLLM_DC) down --remove-orphans

vllm-logs:
	@$(VLLM_DC) logs -f --tail=200
