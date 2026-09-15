import { Router } from "express";
import { Readable } from "node:stream";
import { z } from "zod";

import { logger } from "../logger.js";
import { ragCore, RagCoreError } from "../ragCoreClient.js";

/**
 * POST /query - proxies rag-core's SSE stream through unchanged: real
 * token-by-token text as the answer generates, citations/stop_reason in one
 * final "done" event (no documented per-token citations delta on the
 * Anthropic streaming API - see app/generation/claude.py). This route adds
 * request validation and rate limiting on top; it does not reshape events.
 */

const querySchema = z.object({
  query: z.string().min(1).max(2000),
  top_k: z.number().int().positive().max(50).optional(),
  source_filename: z.string().optional(),
  doc_version: z.string().optional(),
  ingested_after: z.string().optional(),
  ingested_before: z.string().optional(),
  include_superseded: z.boolean().optional(),
  session_id: z.string().max(200).optional(),
  pipeline: z.enum(["improved", "naive"]).optional(),
});

export const queryRouter = Router();

queryRouter.post("/query", async (req, res) => {
  const parsed = querySchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(422).json({ error: "invalid request body", details: parsed.error.flatten() });
    return;
  }

  let upstream: Response;
  try {
    upstream = await ragCore.queryStream(parsed.data);
  } catch (err) {
    logger.error({ event: "query_stream_failed", error: String(err) });
    const status = err instanceof RagCoreError && err.status ? err.status : 502;
    res.status(status).json({ error: String(err) });
    return;
  }

  res.status(200);
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no"); // hint reverse proxies (nginx et al.) not to buffer
  res.flushHeaders();

  if (!upstream.body) {
    res.end();
    return;
  }

  const nodeStream = Readable.fromWeb(upstream.body);
  // Without this listener, an upstream disconnect (rag-core aborting the
  // connection mid-stream - e.g. an unhandled provider error on its side)
  // throws an uncaught 'error' event and crashes the entire process, taking
  // down every other in-flight request too, not just this one. Headers are
  // already sent by this point (SSE requires flushing them before the body
  // streams), so the best a listener can do is end the stream cleanly and
  // tell the client via one more SSE event, if the connection can still take it.
  nodeStream.on("error", (err) => {
    logger.error({ event: "query_stream_upstream_error", error: String(err) });
    if (!res.writableEnded) {
      res.write(`event: error\ndata: ${JSON.stringify({ error: "upstream connection failed" })}\n\n`);
      res.end();
    }
  });
  nodeStream.pipe(res);
  req.on("close", () => nodeStream.destroy());
});
