import request from "supertest";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ragCore } from "../src/ragCoreClient.js";
import { createApp } from "../src/index.js";

// Hermetic: stub the rag-core client so these tests never depend on whether a
// rag-core is actually reachable from the test environment. (vi.mock is hoisted
// above the imports above.)
vi.mock("../src/ragCoreClient.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/ragCoreClient.js")>();
  return {
    ...actual,
    ragCore: { health: vi.fn(), ready: vi.fn(), ping: vi.fn() },
  };
});

describe("api health", () => {
  beforeEach(() => {
    vi.mocked(ragCore.ready).mockReset();
  });

  it("GET /health -> 200 ok", async () => {
    const res = await request(createApp()).get("/health");
    expect(res.status).toBe(200);
    expect(res.body.status).toBe("ok");
    expect(res.body.service).toBe("api");
  });

  it("GET /ready -> 200 when rag-core is ready", async () => {
    vi.mocked(ragCore.ready).mockResolvedValue({ status: "ok" });
    const res = await request(createApp()).get("/ready");
    expect(res.status).toBe(200);
    expect(res.body.status).toBe("ok");
  });

  it("GET /ready -> 503 when rag-core is unreachable", async () => {
    vi.mocked(ragCore.ready).mockRejectedValue(new Error("unreachable"));
    const res = await request(createApp()).get("/ready");
    expect(res.status).toBe(503);
    expect(res.body.status).toBe("degraded");
  });
});
