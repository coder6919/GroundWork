# Architecture

## Services

| Service    | Stack               | Role                                                                 | Public? |
|------------|---------------------|---------------------------------------------------------------------|---------|
| `web`      | React + Vite + Tailwind | Marketing site + demo app. Talks only to `api`. *(Stage 10)*      | yes     |
| `api`      | Node + Express + TS | Public API. Auth, rate limiting, validation, SSE proxying, `/sources`. | yes     |
| `rag-core` | Python + FastAPI    | Ingestion, retrieval, reranking, generation, eval harness.          | **no**  |
| `qdrant`   | Qdrant              | Vector store (dense + sparse). Local: container. Deploy: Qdrant Cloud. | **no**  |

## Request path (target)

```
[Doc sources] -> [Ingestion] -> [Chunking] -> [Embedding] -> [Qdrant]
                                                                 |
[User query] -> web -> api -> rag-core -> [Hybrid retrieval (BM25 + vector)]
                                                                 |
                                        [Rerank top-20 -> top-5]
                                                                 |
                                        [Claude generation w/ citations]
                                                                 |
                                        [Response + sources -> api -> web]
```

## Trust boundary

- Only `api` is intended to be internet-facing.
- `rag-core` and `qdrant` sit on the private `ragnet` network. The base compose
  file exposes **no** host ports for them. `docker-compose.override.yml` adds
  `127.0.0.1`-bound port maps for local debugging only.
- `api -> rag-core` calls carry `X-Internal-Secret: $RAG_CORE_SHARED_SECRET`.
  `rag-core` rejects any internal route without it and **fails closed** when the
  secret is unset. `/health` and `/ready` are the only unauthenticated routes.
- `web` never talks to `rag-core` directly.

## Extensibility seams (reserved, not implemented in v1)

- **Tool interface** in generation: v1 registers exactly one tool
  (`knowledge_search`). Action tools (tickets, CRM, email) slot in later without
  rearchitecting.
- **`tenant_id`** on every stored record, hardcoded `"default"` in v1.
- **Job-based ingestion** so async workers / larger corpora don't need a rewrite.
- **SQLite** document registry + session store behind interfaces, swappable for
  PostgreSQL.

## Stage 0 scope

Skeleton only: the three backend services boot, expose health/readiness probes,
and the `api -> rag-core` shared-secret boundary is enforced and tested. No
chunking, embedding, retrieval, generation, corpus, or provider calls exist yet.
