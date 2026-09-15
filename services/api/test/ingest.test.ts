import request from "supertest";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ragCore } from "../src/ragCoreClient.js";
import { createApp } from "../src/index.js";

// Hermetic: stub the rag-core client so these tests never touch the network.
vi.mock("../src/ragCoreClient.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/ragCoreClient.js")>();
  return {
    ...actual,
    ragCore: { health: vi.fn(), ready: vi.fn(), ping: vi.fn(), ingestUpload: vi.fn(), getSource: vi.fn(), queryStream: vi.fn() },
  };
});

describe("POST /ingest (development - default test env)", () => {
  beforeEach(() => {
    vi.mocked(ragCore.ingestUpload).mockReset();
  });

  it("rejects a request with no file", async () => {
    const res = await request(createApp()).post("/ingest");
    expect(res.status).toBe(400);
  });

  it("rejects an unsupported file extension", async () => {
    const res = await request(createApp())
      .post("/ingest")
      .attach("file", Buffer.from("MZ..."), "payload.exe");
    expect(res.status).toBe(415);
  });

  it("forwards a supported file upload to rag-core and returns its response", async () => {
    vi.mocked(ragCore.ingestUpload).mockResolvedValue({
      root: "policy.md",
      processed: 1,
      ready: 1,
      skipped_duplicate: 0,
      failed: 0,
      unprocessable: 0,
      results: [
        {
          doc_id: "d1",
          filename: "policy.md",
          status: "ready",
          chunk_count: 2,
          failure_reason: null,
          skipped_duplicate: false,
        },
      ],
    });

    const res = await request(createApp())
      .post("/ingest")
      .attach("file", Buffer.from("# Policy\n\nSome content."), "policy.md");

    expect(res.status).toBe(200);
    expect(res.body.ready).toBe(1);
    expect(ragCore.ingestUpload).toHaveBeenCalledTimes(1);
    const call = vi.mocked(ragCore.ingestUpload).mock.calls[0]![0];
    expect(call.originalname).toBe("policy.md");
    expect(call.buffer.toString()).toContain("Some content.");
  });

  it("does not require X-Ingest-Key in development", async () => {
    vi.mocked(ragCore.ingestUpload).mockResolvedValue({
      root: "policy.md",
      processed: 1,
      ready: 1,
      skipped_duplicate: 0,
      failed: 0,
      unprocessable: 0,
      results: [],
    });

    const res = await request(createApp())
      .post("/ingest")
      .attach("file", Buffer.from("# Policy"), "policy.md");

    expect(res.status).toBe(200);
  });
});

describe("POST /ingest (production-like env)", () => {
  const originalEnv = { ...process.env };

  afterEach(() => {
    process.env = { ...originalEnv };
    vi.resetModules();
  });

  it("rejects an upload with no X-Ingest-Key outside development", async () => {
    vi.resetModules();
    process.env.ENV = "production";
    process.env.INGEST_API_KEY = "secret-key";
    const { createApp: freshCreateApp } = await import("../src/index.js");

    const res = await request(freshCreateApp())
      .post("/ingest")
      .attach("file", Buffer.from("# Policy"), "policy.md");

    expect(res.status).toBe(401);
  });

  it("accepts an upload with the correct X-Ingest-Key outside development", async () => {
    vi.resetModules();
    process.env.ENV = "production";
    process.env.INGEST_API_KEY = "secret-key";
    const { createApp: freshCreateApp } = await import("../src/index.js");
    const { ragCore: freshRagCore } = await import("../src/ragCoreClient.js");
    vi.mocked(freshRagCore.ingestUpload).mockResolvedValue({
      root: "policy.md",
      processed: 1,
      ready: 1,
      skipped_duplicate: 0,
      failed: 0,
      unprocessable: 0,
      results: [],
    });

    const res = await request(freshCreateApp())
      .post("/ingest")
      .set("X-Ingest-Key", "secret-key")
      .attach("file", Buffer.from("# Policy"), "policy.md");

    expect(res.status).toBe(200);
  });

  it("returns 403 when ingestion is disabled", async () => {
    vi.resetModules();
    process.env.INGEST_ENABLED = "false";
    const { createApp: freshCreateApp } = await import("../src/index.js");

    const res = await request(freshCreateApp())
      .post("/ingest")
      .attach("file", Buffer.from("# Policy"), "policy.md");

    expect(res.status).toBe(403);
  });
});
