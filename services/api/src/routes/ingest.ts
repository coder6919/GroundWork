import { Router } from "express";
import multer from "multer";

import { config } from "../config.js";
import { logger } from "../logger.js";
import { ragCore, RagCoreError } from "../ragCoreClient.js";

/**
 * POST /ingest - the only ingest route this public layer exposes. Takes a
 * file upload (multipart), never a server-side path: rag-core's
 * /internal/ingest is never reachable from the internet, and this route must
 * not become a way to point ingestion at an arbitrary path inside it (see
 * DIRECTION.md's hosted-demo rule - "never an unrestricted public ingestion
 * endpoint").
 *
 * Gated by INGEST_ENABLED and, outside development, an INGEST_API_KEY sent
 * as the X-Ingest-Key header - separate from RAG_CORE_SHARED_SECRET, which
 * protects the api -> rag-core boundary, not the public -> api one.
 */

const ALLOWED_EXTENSIONS = new Set([".txt", ".md", ".markdown", ".pdf"]);

function hasAllowedExtension(filename: string): boolean {
  const dot = filename.lastIndexOf(".");
  if (dot < 0) return false;
  return ALLOWED_EXTENSIONS.has(filename.slice(dot).toLowerCase());
}

export const ingestRouter = Router();

function buildUpload() {
  return multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: config.MAX_UPLOAD_BYTES, files: 1 },
  });
}

ingestRouter.post("/ingest", (req, res, next) => {
  buildUpload().single("file")(req, res, (err: unknown) => {
    if (err) {
      if (err instanceof multer.MulterError && err.code === "LIMIT_FILE_SIZE") {
        res.status(413).json({ error: `file exceeds the ${config.MAX_UPLOAD_BYTES}-byte limit` });
        return;
      }
      res.status(400).json({ error: String(err) });
      return;
    }
    next();
  });
});

ingestRouter.post("/ingest", async (req, res) => {
  if (!config.INGEST_ENABLED) {
    res.status(403).json({ error: "ingestion is disabled" });
    return;
  }

  if (config.ENV !== "development") {
    const key = req.header("X-Ingest-Key");
    if (!config.INGEST_API_KEY || key !== config.INGEST_API_KEY) {
      res.status(401).json({ error: "invalid or missing X-Ingest-Key" });
      return;
    }
  }

  if (!req.file) {
    res.status(400).json({ error: 'no file uploaded (expected multipart field "file")' });
    return;
  }
  if (!hasAllowedExtension(req.file.originalname)) {
    res.status(415).json({ error: `unsupported file type: ${req.file.originalname}` });
    return;
  }

  try {
    const result = await ragCore.ingestUpload({
      buffer: req.file.buffer,
      originalname: req.file.originalname,
      mimetype: req.file.mimetype,
    });
    res.json(result);
  } catch (err) {
    logger.error({ event: "ingest_upload_failed", error: String(err) });
    const status = err instanceof RagCoreError && err.status ? err.status : 502;
    res.status(status).json({ error: String(err) });
  }
});
