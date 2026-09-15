SHELL := /bin/bash
COMPOSE := docker compose

.PHONY: help up down logs ps health test test-rag test-api test-integration verify lock clean eval eval-naive

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

up: ## Build and start all services (detached)
	$(COMPOSE) up --build -d

down: ## Stop and remove containers
	$(COMPOSE) down

logs: ## Tail service logs
	$(COMPOSE) logs -f --tail=100

ps: ## Show service status
	$(COMPOSE) ps

health: ## Run host-side health checks against a running stack
	bash ./scripts/healthcheck.sh

test: test-rag test-api ## Run unit test suites (no network, no paid APIs)

test-rag: ## rag-core unit tests (pytest, one-off dev container)
	$(COMPOSE) run --rm --no-deps rag-core uv run --frozen pytest -v -m "not integration"

test-api: ## api unit tests (vitest, one-off dev container)
	$(COMPOSE) run --rm --no-deps api npm test

test-integration: ## Integration tests against a running stack (run `make up` first)
	$(COMPOSE) run --rm --no-deps -e RUN_INTEGRATION=1 rag-core uv run --frozen pytest -v -m integration
	$(COMPOSE) run --rm --no-deps -e RUN_INTEGRATION=1 api npm test

verify: ## Full Stage-0 verification: up -> wait -> health -> unit -> integration
	$(COMPOSE) up --build -d
	bash ./scripts/wait-for-health.sh
	bash ./scripts/healthcheck.sh
	$(MAKE) test
	$(MAKE) test-integration

eval: ## Run the Stage 8 golden eval (improved pipeline by default). Ingests corpus/files + corpus/sample first (idempotent). Explicitly triggered - makes real paid API calls (Voyage/Cohere/Anthropic), never run in CI.
	@echo "Ingesting real + sample corpus (skips anything already unchanged)..."
	@SECRET=$$(grep -E '^RAG_CORE_SHARED_SECRET=' .env | cut -d= -f2); \
	curl -sf -X POST http://localhost:8000/internal/ingest -H "X-Internal-Secret: $$SECRET" -H "Content-Type: application/json" -d '{"path": "/corpus/files"}' > /dev/null; \
	curl -sf -X POST http://localhost:8000/internal/ingest -H "X-Internal-Secret: $$SECRET" -H "Content-Type: application/json" -d '{"path": "/corpus/sample"}' > /dev/null
	$(COMPOSE) run --rm --no-deps rag-core uv run --frozen python -m scripts.run_eval $(ARGS)

eval-naive: ## Run the Stage 8 golden eval against the naive baseline pipeline.
	$(MAKE) eval ARGS="--pipeline naive $(ARGS)"

lock: ## Regenerate the rag-core uv lockfile
	cd services/rag-core && uv lock

clean: ## Remove containers, volumes and build cache
	$(COMPOSE) down -v --remove-orphans
