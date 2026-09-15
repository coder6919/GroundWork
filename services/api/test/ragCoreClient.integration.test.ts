import { describe, expect, it } from "vitest";

import { ragCore } from "../src/ragCoreClient.js";

const run = process.env.RUN_INTEGRATION === "1";

describe.skipIf(!run)("ragCoreClient (integration)", () => {
  it("health() reaches rag-core", async () => {
    const h = await ragCore.health();
    expect(h.status).toBe("ok");
  });

  it("ping() authenticates with the shared secret", async () => {
    const p = await ragCore.ping();
    expect(p.authenticated).toBe(true);
  });
});
