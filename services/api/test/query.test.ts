import request from "supertest";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ragCore } from "../src/ragCoreClient.js";
import { createApp } from "../src/index.js";

vi.mock("../src/ragCoreClient.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/ragCoreClient.js")>();
  return {
    ...actual,
    ragCore: { health: vi.fn(), ready: vi.fn(), ping: vi.fn(), ingestUpload: vi.fn(), getSource: vi.fn(), queryStream: vi.fn() },
  };
});

function fakeSseResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(stream, { status: 200 });
}

describe("POST /query", () => {
  beforeEach(() => {
    vi.mocked(ragCore.queryStream).mockReset();
  });

  it("rejects an empty query", async () => {
    const res = await request(createApp()).post("/query").send({ query: "" });
    expect(res.status).toBe(422);
    expect(ragCore.queryStream).not.toHaveBeenCalled();
  });

  it("rejects a missing query field", async () => {
    const res = await request(createApp()).post("/query").send({});
    expect(res.status).toBe(422);
  });

  it("proxies rag-core's SSE stream through unchanged", async () => {
    vi.mocked(ragCore.queryStream).mockResolvedValue(
      fakeSseResponse([
        'event: meta\ndata: {"session_id":null,"rewritten_query":null,"retrieved_count":2}\n\n',
        'event: token\ndata: {"text":"Hello "}\n\n',
        'event: done\ndata: {"text":"Hello world.","citations":[],"model":"claude-sonnet-5","stop_reason":"end_turn"}\n\n',
      ]),
    );

    const res = await request(createApp()).post("/query").send({ query: "hi" });

    expect(res.status).toBe(200);
    expect(res.headers["content-type"]).toMatch(/text\/event-stream/);
    expect(res.text).toContain("event: meta");
    expect(res.text).toContain("event: token");
    expect(res.text).toContain("event: done");
    expect(ragCore.queryStream).toHaveBeenCalledWith(expect.objectContaining({ query: "hi" }));
  });

  it("returns 502 when rag-core is unreachable", async () => {
    const { RagCoreError } = await import("../src/ragCoreClient.js");
    vi.mocked(ragCore.queryStream).mockRejectedValue(new RagCoreError("rag-core unreachable"));

    const res = await request(createApp()).post("/query").send({ query: "hi" });

    expect(res.status).toBe(502);
  });

  it("ends the stream with an error event instead of crashing when the upstream connection breaks mid-stream", async () => {
    // Live incident: rag-core aborted an already-started SSE stream (its own
    // unhandled exception after sending a 200), and the previous version of
    // this route had no error listener on the piped Node stream - an
    // unhandled 'error' event there crashes the entire process, taking down
    // every other in-flight request too. If that regressed, this test
    // itself would crash the vitest worker rather than fail normally.
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.error(new Error("terminated"));
      },
    });
    vi.mocked(ragCore.queryStream).mockResolvedValue(new Response(stream, { status: 200 }));

    const res = await request(createApp()).post("/query").send({ query: "hi" });

    expect(res.status).toBe(200); // headers were already sent before the break
    expect(res.text).toContain("event: error");
    expect(res.text).toContain("upstream connection failed");
  });
});

describe("POST /query rate limiting", () => {
  it("returns 429 once the per-window limit is exceeded", async () => {
    vi.resetModules();
    process.env.RATE_LIMIT_MAX_QUERY = "1";
    const { createApp: freshCreateApp } = await import("../src/index.js");
    const { ragCore: freshRagCore } = await import("../src/ragCoreClient.js");
    vi.mocked(freshRagCore.queryStream).mockResolvedValue(fakeSseResponse(["event: done\ndata: {}\n\n"]));

    const app = freshCreateApp();
    const first = await request(app).post("/query").send({ query: "hi" });
    const second = await request(app).post("/query").send({ query: "hi again" });

    expect(first.status).toBe(200);
    expect(second.status).toBe(429);

    delete process.env.RATE_LIMIT_MAX_QUERY;
    vi.resetModules();
  });
});
