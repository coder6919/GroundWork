import { Router } from "express";

import { logger } from "../logger.js";
import { ragCore, RagCoreError } from "../ragCoreClient.js";

/** GET /sources/:id - registry metadata for citation click-through (Stage 9).
 * Metadata only, not the original file bytes - serving/previewing those is a
 * Stage 10 concern if the frontend actually needs it. */
export const sourcesRouter = Router();

sourcesRouter.get("/sources/:id", async (req, res) => {
  const { id } = req.params;
  try {
    const source = await ragCore.getSource(id);
    res.json(source);
  } catch (err) {
    logger.error({ event: "get_source_failed", id, error: String(err) });
    const status = err instanceof RagCoreError && err.status ? err.status : 502;
    res.status(status).json({ error: String(err) });
  }
});
