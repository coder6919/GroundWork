#!/usr/bin/env bash
# Host-side health checks for the Stage 0 stack.
# Uses the dev-only localhost port maps from docker-compose.override.yml.
# Each check is retried briefly so a slow-starting container is not a false failure.
set -uo pipefail

QDRANT_URL="${QDRANT_HOST_URL:-http://127.0.0.1:6333}"
RAG_CORE_URL="${RAG_CORE_HOST_URL:-http://127.0.0.1:8000}"
API_URL="${API_HOST_URL:-http://127.0.0.1:8080}"
RETRIES="${HEALTHCHECK_RETRIES:-15}"
SLEEP="${HEALTHCHECK_SLEEP:-2}"

# Load RAG_CORE_SHARED_SECRET from .env if present and not already in the shell.
if [[ -z "${RAG_CORE_SHARED_SECRET:-}" && -f .env ]]; then
  RAG_CORE_SHARED_SECRET="$(grep -E '^RAG_CORE_SHARED_SECRET=' .env | head -1 | cut -d= -f2- || true)"
fi

fail=0

http_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$@" 2>/dev/null || echo "000"; }

# check <name> <expected> <curl-args...>
check() {
  local name="$1" want="$2"; shift 2
  local got=""
  for ((i = 1; i <= RETRIES; i++)); do
    got="$(http_code "$@")"
    [[ "$got" == "$want" ]] && { printf '  OK    %-46s -> %s\n' "$name" "$got"; return 0; }
    sleep "$SLEEP"
  done
  printf '  FAIL  %-46s -> %s (expected %s)\n' "$name" "$got" "$want"
  fail=1
}

echo "Health checks (up to $((RETRIES * SLEEP))s per check):"
check "qdrant   GET /readyz"                  200 "$QDRANT_URL/readyz"
check "rag-core GET /health"                  200 "$RAG_CORE_URL/health"
check "rag-core GET /ready (Qdrant wired)"    200 "$RAG_CORE_URL/ready"
check "api      GET /health"                  200 "$API_URL/health"
check "api      GET /ready (aggregated)"      200 "$API_URL/ready"

echo "Internal shared-secret auth:"
check "rag-core /internal/ping  no secret"    401 "$RAG_CORE_URL/internal/ping"
check "rag-core /internal/ping  wrong secret" 401 -H 'X-Internal-Secret: wrong' "$RAG_CORE_URL/internal/ping"

if [[ -n "${RAG_CORE_SHARED_SECRET:-}" ]]; then
  check "rag-core /internal/ping  valid secret" 200 -H "X-Internal-Secret: ${RAG_CORE_SHARED_SECRET}" "$RAG_CORE_URL/internal/ping"
  check "api      /debug/rag-core-ping (e2e)"   200 "$API_URL/debug/rag-core-ping"
else
  echo "  SKIP  valid-secret checks (RAG_CORE_SHARED_SECRET not found in shell or .env)"
fi

echo
if [[ "$fail" -eq 0 ]]; then echo "All health checks passed."; else echo "Some health checks FAILED."; fi
exit $fail
