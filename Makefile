.PHONY: help quickstart quickstart-ml up down logs test \
        ml-up ml-down ml-logs ml-list ml-fetch ensure-inference-token

help:
	@echo "Core stack (rules only — no ML, nothing downloaded):"
	@echo "  make quickstart     build + start + wait for healthy + fire a sample scan"
	@echo "  make up             docker compose up -d"
	@echo "  make down           docker compose down"
	@echo "  make logs           docker compose logs -f armor-core"
	@echo "  make test           cargo test --workspace"
	@echo ""
	@echo "ML tier (optional, additive — see inference/README.md):"
	@echo "  make quickstart-ml  build + start the FULL stack incl. inference,"
	@echo "                      fetch a real model (default: prompt_injection,"
	@echo "                      override with ML_TASK=<task>), wait for both"
	@echo "                      services healthy, fire a sample scan — the"
	@echo "                      one-command path to a real ML-backed verdict"
	@echo "  make ml-up          start the sidecar on stub runners, no downloads"
	@echo "  make ml-down        stop the sidecar"
	@echo "  make ml-logs        docker compose logs -f inference"
	@echo "  make ml-list        list the vetted models per task"
	@echo "  make ml-fetch TASK=prompt_injection"
	@echo "                      fetch + export + quantize a real model into the"
	@echo "                      shared volume, no local Python/torch install"
	@echo "                      required; prints the sha256 to pin"

quickstart:
	@./scripts/quickstart.sh

quickstart-ml: ensure-inference-token
	@./scripts/quickstart-ml.sh

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f armor-core

test:
	cargo test --workspace

# `install`/`reload` on the sidecar always require a bearer token (see
# inference/src/armor_inference/main.py's `require_mutation_token`) — with
# none configured, it mints a random one at boot and only logs it, which
# armor-core has no way to read, so every install call 401s. Generate a
# shared dev-only secret into .env (gitignored) the first time `ml-up` runs,
# so `ARMOR_INFERENCE_AUTH_TOKEN` — read by both services via
# docker-compose.yml's `${ARMOR_INFERENCE_AUTH_TOKEN:-}` — is already set
# before either container starts. Idempotent: leaves an existing token
# alone, so it doesn't get rotated (and every already-running container
# desynced from it) on every `make ml-up`.
ensure-inference-token:
	@if ! grep -qs '^ARMOR_INFERENCE_AUTH_TOKEN=.' .env 2>/dev/null; then \
		printf 'ARMOR_INFERENCE_AUTH_TOKEN=%s\n' "$$(openssl rand -hex 24)" >> .env; \
		echo "==> Generated a dev-only ARMOR_INFERENCE_AUTH_TOKEN in .env (gitignored)"; \
		echo "    — shared secret between armor-core and the inference sidecar,"; \
		echo "    so model installs via /api/v1/models/install work out of the box."; \
	fi

ml-up: ensure-inference-token
	docker compose --profile ml up -d

ml-down:
	docker compose --profile ml down

ml-logs:
	docker compose logs -f inference

ml-list:
	docker compose --profile ml-fetch run --rm ml-fetch --list

ml-fetch:
	@if [ -z "$(TASK)" ]; then \
		echo "usage: make ml-fetch TASK=<task-name>   (see: make ml-list)" >&2; \
		exit 2; \
	fi
	docker compose --profile ml-fetch run --rm ml-fetch --task $(TASK)
