#!/usr/bin/env bash
# Block until every compose service with a healthcheck reports "healthy",
# or fail after a timeout. Used by `make verify` and CI.
set -uo pipefail

TIMEOUT="${HEALTH_TIMEOUT:-120}"
INTERVAL=3
elapsed=0

services=$(docker compose ps --services 2>/dev/null)

while (( elapsed < TIMEOUT )); do
  all_ok=1
  for svc in $services; do
    cid=$(docker compose ps -q "$svc" 2>/dev/null)
    [[ -z "$cid" ]] && { all_ok=0; continue; }
    status=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$cid" 2>/dev/null)
    case "$status" in
      healthy|running) ;;
      *) all_ok=0 ;;
    esac
  done
  if (( all_ok == 1 )); then
    echo "All services healthy after ${elapsed}s."
    exit 0
  fi
  sleep "$INTERVAL"
  (( elapsed += INTERVAL ))
done

echo "Timed out after ${TIMEOUT}s waiting for services to become healthy."
docker compose ps
docker compose logs --tail=50
exit 1
