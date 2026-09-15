import express from "express";
import rateLimit from "express-rate-limit";
import { pinoHttp } from "pino-http";

import { config, configWarnings } from "./config.js";
import { logger } from "./logger.js";
import { ragCore, RagCoreError } from "./ragCoreClient.js";
import { ingestRouter } from "./routes/ingest.js";
import { queryRouter } from "./routes/query.js";
import { sourcesRouter } from "./routes/sources.js";

/**
 * Public surface:
 *   GET  /health              - liveness
 *   GET  /ready                - aggregated readiness (api is ready iff rag-core is)
 *   GET  /debug/rag-core-ping  - scaffolding: proves api can authenticate to rag-core
 *   POST /query    (Stage 9)  - SSE proxy of rag-core's streamed answer, rate-limited
 *   POST /ingest   (Stage 9)  - file upload only (never a server-side path), gated
 *                               by INGEST_ENABLED + X-Ingest-Key outside development
 *   GET  /sources/:id (Stage 9) - registry metadata for citation click-through
 */
export function createApp() {
  const app = express();
  app.use(pinoHttp({ logger }));
  app.use(express.json({ limit: "1mb" }));

  // Stage 10: the web frontend runs on its own origin. A named allow (not a
  // wildcard) since X-Ingest-Key must never be exposed cross-origin to an
  // unapproved site.
  app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", config.WEB_ORIGIN);
    res.header("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
    res.header("Access-Control-Allow-Headers", "Content-Type, X-Ingest-Key");
    if (req.method === "OPTIONS") {
      res.sendStatus(204);
      return;
    }
    next();
  });

  app.get("/health", (_req, res) => {
    res.json({ status: "ok", service: "api", version: "0.0.0" });
  });

  app.get("/ready", async (_req, res) => {
    try {
      const ragCoreStatus = await ragCore.ready();
      res.json({ status: "ok", ragCore: ragCoreStatus });
    } catch (err) {
      res.status(503).json({ status: "degraded", error: String(err) });
    }
  });

  app.get("/debug/rag-core-ping", async (_req, res) => {
    try {
      const result = await ragCore.ping();
      res.json({ status: "ok", ragCore: result });
    } catch (err) {
      const code = err instanceof RagCoreError && err.status ? err.status : 502;
      res.status(code).json({ status: "error", error: String(err) });
    }
  });

  // In-memory, per-instance rate limiting - appropriate for v1 (no
  // horizontal scaling; see DIRECTION.md's "Out of v1" list). /query and
  // /ingest are the only routes that spend real money downstream, so they're
  // the only ones limited.
  app.use(
    "/query",
    rateLimit({
      windowMs: config.RATE_LIMIT_WINDOW_MS,
      limit: config.RATE_LIMIT_MAX_QUERY,
      standardHeaders: true,
      legacyHeaders: false,
    }),
  );
  app.use(
    "/ingest",
    rateLimit({
      windowMs: config.RATE_LIMIT_WINDOW_MS,
      limit: config.RATE_LIMIT_MAX_INGEST,
      standardHeaders: true,
      legacyHeaders: false,
    }),
  );

  app.use(queryRouter);
  app.use(ingestRouter);
  app.use(sourcesRouter);

  return app;
}

const isTest = process.env.VITEST === "true" || process.env.NODE_ENV === "test";
if (!isTest) {
  for (const warning of configWarnings()) logger.warn(warning);
  createApp().listen(config.API_PORT, () => {
    logger.info(
      { env: config.ENV, port: config.API_PORT, ragCoreUrl: config.RAG_CORE_URL },
      "api listening",
    );
  });
}
