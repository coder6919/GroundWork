import request from "supertest";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ragCore, RagCoreError } from "../src/ragCoreClient.js";
import { createApp } from "../src/index.js";

vi.mock("../src/ragCoreClient.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/ragCoreClient.js")>();
  return {
    ...actual,
    ragCore: { health: vi.fn(), ready: vi.fn(), ping: vi.fn(), ingestUpload: vi.fn(), getSource: vi.fn(), queryStream: vi.fn() },
  };
});

describe("GET /sources/:id", () => {
  beforeEach(() => {
    vi.mocked(ragCore.getSource).mockReset();
  });

  it("returns the registry metadata for a known id", async () => {
    vi.mocked(ragCore.getSource).mockResolvedValue({
      doc_id: "abc123",
      filename: "refund-policy.md",
      mime_type: "text/markdown",
      status: "ready",
      page_count: null,
      version: null,
      supersedes: null,
      chunk_count: 6,
      ingested_at: "2026-01-01T00:00:00+00:00",
    });

    const res = await request(createApp()).get("/sources/abc123");

    expect(res.status).toBe(200);
    expect(res.body.doc_id).toBe("abc123");
    expect(res.body.chunk_count).toBe(6);
    expect(ragCore.getSource).toHaveBeenCalledWith("abc123");
  });

  it("returns 404 when rag-core reports an unknown id", async () => {
    vi.mocked(ragCore.getSource).mockRejectedValue(new RagCoreError("rag-core /internal/sources/nope responded 404", 404));

    const res = await request(createApp()).get("/sources/nope");

    expect(res.status).toBe(404);
  });
});
